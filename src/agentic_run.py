import json
import csv
import os
import argparse
from datetime import datetime
from typing import TypedDict, Annotated, List, Union, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

from src.models.product import Product
from src.models.vending_machine import VendingMachine
from src.simulation.engine import simulate_competitive_week
from src.llm.client import OpenRouterClient
from src.llm.tools import VendingMachineTools
from src.config import (
    DEFAULT_MODEL, 
    COMPETITION_PRODUCT_CONFIGS, 
    MAINTENANCE_COST, 
    NUM_WEEKS, 
    SIMULATION_RESULTS_CSV,
    OPENROUTER_API_KEY,
    IMAGES_DIR,
    LOGS_DIR,
    RESULTS_DIR
)
from src.utils.helpers import plot_profits

# --- 1. Define State ---
# --- 1. Define State ---
class AgentState(TypedDict):
    week: int
    sim_weeks: int
    messages: Annotated[List[BaseMessage], operator.add]
    market_data: Dict[str, Any]
    stockout_events: List[str] # This is per-week, so overwrite is actually OK/desired? No, simulate returns it.
    # Wait, stockout_events is used in 'model' to show events causing current state. 
    # If we accumulate, we show ALL history events? 
    # The existing code in 'simulate' returns "stockout_events": stockout_events.
    # The prompt generator uses: event_feedback = "\n".join(stockout_events)
    # If we accumulate, event_feedback becomes huge.
    # So stockout_events SHOULD overwrite (be per step).
    
    basic_profit: float
    llm_profit: float
    
    # These MUST accumulate for the final plots/logs:
    weekly_stats: Annotated[List[Dict[str, Any]], operator.add]
    weeks_list: Annotated[List[int], operator.add]
    basic_profits: Annotated[List[float], operator.add]
    llm_profits: Annotated[List[float], operator.add]
    
    machines: Dict[str, VendingMachine]
    tools_instance: VendingMachineTools
    verbose: bool
    target_model: str

# --- 2. Define Tools Wrapper ---
# We need to wrap the VendingMachineTools for LangChain
# Since VendingMachineTools methods are instance methods, we'll create a helper wrapper
# inside the node or purely functional tools that access a global/passed instance.
# For simplicity, we'll define the tools within the "Act" node or using bind_tools if we had a pure model.
# But here, we want the agent to call ANY tool.

# Actually, the best way with LangGraph and existing classes is to create a "ToolNode" that executes the
# requested tool. The model will just output the JSON (as in the original script) or we can use function calling.
# Given the user wants to "use the same prompt", we should probably stick to the existing text-based JSON interface
# OR upgrade to function calling if the "best agent framework" implies it.
# The prompt explicitly asks for JSON. Let's stick to the prompt structure but let the agent "Reason" before "Acting".

def run_simulation_step(state: AgentState):
    """
    Simulates one week of the competition and updates the state.
    """
    week = state['week']
    sim_weeks = state['sim_weeks']
    basic_machine = state['machines']['BasicMachine']
    llm_machine = state['machines']['LLMMachine']
    
    if week > sim_weeks:
        return {"messages": [SystemMessage(content="Simulation Finished")]}

    # Simulate the week
    results, stockout_events = simulate_competitive_week([basic_machine, llm_machine], week)
    
    # Calculate profits
    b_profit = basic_machine.calculate_profit_loss()
    l_profit = llm_machine.calculate_profit_loss()
    
    # Stats
    b_avg_price = sum(p.price for p in basic_machine.products.values()) / len(basic_machine.products)
    l_avg_price = sum(p.price for p in llm_machine.products.values()) / len(llm_machine.products)
    b_avg_stock = sum(p.stock for p in basic_machine.products.values()) / len(basic_machine.products)
    l_avg_stock = sum(p.stock for p in llm_machine.products.values()) / len(llm_machine.products)
    
    weekly_stat = {
        "week": week,
        "basic_profit": f"{b_profit:.2f}",
        "llm_profit": f"{l_profit:.2f}",
        "basic_avg_price": f"{b_avg_price:.2f}",
        "llm_avg_price": f"{l_avg_price:.2f}",
        "basic_avg_stock": f"{b_avg_stock:.2f}",
        "llm_avg_stock": f"{l_avg_stock:.2f}"
    }
    
    # Market Data for next step
    tools_instance = state['tools_instance']
    market_data = tools_instance.get_market_data()
    
    event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
    
    if state['verbose']:
        print(f"--- Week {week} Complete ---")
        print(f"Basic Profit: {b_profit:.2f}, LLM Profit: {l_profit:.2f}")

    return {
        "stockout_events": stockout_events,
        "basic_profit": b_profit,
        "llm_profit": l_profit,
        "market_data": market_data,
        "weeks_list": [week],
        "basic_profits": [b_profit],
        "llm_profits": [l_profit],
        "weekly_stats": [weekly_stat]
    }

def model_node(state: AgentState):
    """
    The agent reasons about the current state and decides on actions.
    """
    week = state['week']
    sim_weeks = state['sim_weeks']
    market_data = state['market_data']
    stockout_events = state['stockout_events']
    
    event_feedback = "\n".join(stockout_events) if stockout_events else "No products ran out of stock this week."
    
    system_prompt = f"""
    You are a Strategic Business Manager for 'LLMMachine'. You are in direct competition with 'BasicMachine'.
    Both machines share the same client pool. Clients choose the product with the highest 'purchase_likelihood'.

    IMPORTANT RULES:
    1. AT THE START OF EACH WEEK, YOUR MACHINE IS REFILLED TO MAX CAPACITY.
    2. 'BasicMachine' NEVER changes its prices.
    3. Your goal is to maximize NET PROFIT.

    STRATEGIC ADVICE:
    - Your priority is PROFIT, not just sales volume.
    - Selling out is NOT always the goal. If there is more supply than demand, you might never sell out.
    - If you raise prices, you might sell fewer items, but if the margin increase covers the volume loss, your PROFIT goes up.
    - BasicMachine prices are static. Use this to your advantage.
    - Monitor your Weekly Profit. If a price hike increased profit, keep it or hike further.
    
    AVAILABLE TOOLS (Respond only in JSON):
    1. {{"action": "change_price", "parameters": {{"machine_name": "LLMMachine", "product_name": "...", "new_price": ...}}}}
    2. {{"action": "next_week"}} - to proceed.
    
    Provide your reasoning first, then the JSON block.
    """
    
    is_last_week = (week >= sim_weeks)
    
    if is_last_week:
        prompt = f"""
        --- FINAL WEEK {week} RESULTS ---
        Stockout Events:
        {event_feedback}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        NOTE: This was the FINAL week of the simulation. No further actions can be taken. 
        Please provide your final analysis of the competition. 
        (Do NOT use 'next_week').
        """
    else:
        prompt = f"""
        --- WEEK {week} RESULTS ---
        Stockout Events:
        {event_feedback}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        Machines are being refilled NOW. What adjustments will you make for Week {week+1}?
        """

    # Call LLM
    # We use ChatOpenAI client wrapper to interface with OpenRouter
    llm = ChatOpenAI(
        model=state['target_model'],
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/santos-sanz/LLM_Vending_Machine",
            "X-Title": "LLM Vending Machine Simulation",
        }
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    if state['verbose']:
        print(f"\n[Agent] Thinking for Week {week}...")
        
    response = llm.invoke(messages)
    
    if state['verbose']:
        print(f"[Agent] Response: {response.content}")
        
    return {"messages": [response]}

def tools_node(state: AgentState):
    """
    Parses the LLM response and executes tools.
    """
    last_message = state['messages'][-1]
    content = last_message.content
    tools_instance = state['tools_instance']
    verbose = state['verbose']
    
    # helper from original script
    def extract_json_objects(text):
        json_objects = []
        decoder = json.JSONDecoder()
        pos = 0
        while True:
            match = text.find('{', pos)
            if match == -1:
                break
            try:
                result, index = decoder.raw_decode(text[match:])
                if "action" in result:
                    json_objects.append(result)
                pos = match + index
            except ValueError:
                pos = match + 1
        return json_objects

    commands = extract_json_objects(content)
    
    if not commands:
        # No commands found or just text. If it's the last week, we end. 
        # If not, we might defaulting to next_week or just assume the agent is chatting.
        # But to keep simulation moving, if no "next_week" is strictly found but commands are missing,
        # we might be stuck. Let's force next_week if no valid action commands are found but we need to proceed.
        # However, purely relying on the agent to say "next_week" is safer for the "same prompt" requirement.
        pass

    proceed_to_next_week = False
    
    for command in commands:
        action = command.get("action")
        if action == "change_price":
            params = command.get("parameters", {})
            if isinstance(params, list):
                for p in params:
                    try:
                        res = tools_instance.change_price(**p)
                        if verbose: print(f"Executed: {res}")
                    except Exception as e:
                        print(f"Error executing change_price: {e}")
            else:
                try:
                    res = tools_instance.change_price(**params)
                    if verbose: print(f"Executed: {res}")
                except Exception as e:
                    print(f"Error executing change_price: {e}")
            
            # Legacy behavior: Any action implies end of turn for the week.
            proceed_to_next_week = True
            
        elif action == "next_week":
            proceed_to_next_week = True
            
    # Also, if no commands were found but the model is just chatting?
    # Legacy script breaks loop if `not commands` implies we are done?
    # No, legacy loop: `if not commands: break`.
    # So actually, if the model output is just text (no JSON), the legacy script ALSO proceeds to next week!
    # So we should default proceed_to_next_week = True unless we want multi-turn reasoning.
    # Given the prompt structure, let's enforce single-turn per week to match legacy exactly.
    if not commands:
         proceed_to_next_week = True
    
    if proceed_to_next_week:
        return {"week": state['week'] + 1}
    
    return {} # Changes only applied to machine state, not graph state except maybe messages

def should_continue(state: AgentState):
    """
    Decides whether to go to 'simulate' (next week), 'act' (tools), or END.
    """
    week = state['week']
    sim_weeks = state['sim_weeks']
    last_message = state['messages'][-1]
    content = last_message.content
    
    # week: The current week the agent is *thinking* about or has just simulated.
    # weeks_list: The list of *totally completed* simulation steps.
    
    completed_weeks = len(state['weeks_list'])
    
    # 1. Check if we need to run the simulation for the current 'week'.
    # If tools updated 'week' to X, and we only have X-1 in weeks_list, we must simulate X.
    if week > completed_weeks:
        # Safety check: if we are trying to start a week beyond the limit, we stop.
        if week > sim_weeks:
             return "end"
        return "simulate"

    # 2. If we are here, it means the simulation for 'week' has ALREADY happened.
    # We are now in the loop: "model" -> "tools" -> "check".
    
    # If 'week' is the final week, and we just came from 'tools' (processing the model's final words),
    # we should end. The model has given its final analysis.
    if week >= sim_weeks:
        return "end"
        
    # 3. If it's not the final week, and tools didn't advance the week (no "next_week"),
    # we might technically loop back to "model". 
    # However, in the original script, if the model just changes prices and doesn't say next_week,
    # the loop continues, prompting with updated market data.
    # So we go back to "model".
    
    return "model"

def extract_json_objects(text):
    json_objects = []
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1:
            break
        try:
            result, index = decoder.raw_decode(text[match:])
            if "action" in result:
                json_objects.append(result)
            pos = match + index
        except ValueError:
            pos = match + 1
    return json_objects


# --- 3. Build Graph ---
builder = StateGraph(AgentState)

builder.add_node("simulate", run_simulation_step)
builder.add_node("model", model_node)
builder.add_node("tools", tools_node)

builder.set_entry_point("simulate")

builder.add_edge("simulate", "model")
builder.add_edge("model", "tools")

builder.add_conditional_edges(
    "tools",
    should_continue,
    {
        "simulate": "simulate",
        "model": "model",
        "end": END
    }
)

graph = builder.compile()

# --- 4. Main Execution Wrapper ---
def run_agentic_competition(model_name=None, num_weeks=None, verbose=True):
    target_model = model_name or DEFAULT_MODEL
    sim_weeks = num_weeks or NUM_WEEKS
    
    if verbose:
        print(f"--- Starting Agentic Competition: BasicMachine vs {target_model} ({sim_weeks} weeks) ---\n")

    # Initialize Machines
    basic_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)
    llm_machine = VendingMachine(initial_cash=0, maintenance_cost=MAINTENANCE_COST, initial_investment=0.0)

    for config in COMPETITION_PRODUCT_CONFIGS:
        basic_machine.add_product(Product(**config))
        llm_machine.add_product(Product(**config))

    machines = {
        "BasicMachine": basic_machine,
        "LLMMachine": llm_machine
    }
    
    tools_instance = VendingMachineTools(machines)
    
    # Initial State
    initial_state = AgentState(
        week=1,
        sim_weeks=sim_weeks,
        messages=[],
        market_data={},
        stockout_events=[],
        basic_profit=0.0,
        llm_profit=0.0,
        weekly_stats=[],
        weeks_list=[],
        basic_profits=[],
        llm_profits=[],
        machines=machines,
        tools_instance=tools_instance,
        verbose=verbose,
        target_model=target_model
    )
    
    # Run Graph
    # Default recursion limit is 25, which is too low for multi-week simulations.
    # Each week takes ~3 steps. So we need at least weeks * 3. 
    # We'll set a safe buffer.
    recursion_limit = max(100, sim_weeks * 10)
    final_state = graph.invoke(initial_state, {"recursion_limit": recursion_limit})
    
    # Extract Results
    basic_final = final_state['basic_profit']
    llm_final = final_state['llm_profit']
    weekly_stats = final_state['weekly_stats']
    weeks_list = final_state['weeks_list']
    basic_profits = final_state['basic_profits']
    llm_profits = final_state['llm_profits']

    if verbose:
        print("\n=== Agentic Competition Finished ===")
        print(f"BasicMachine Final Profit: {basic_final:.2f}")
        print(f"LLMMachine ({target_model}) Final Profit: {llm_final:.2f}")

    # Persistence & Plotting (Copied from competitive_run)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = target_model.replace("/", "_").replace(":", "_")
    graph_filename = f"profit_comparison_agentic_{safe_model_name}_{timestamp}.png"
    graph_path = os.path.join(IMAGES_DIR, graph_filename)
    
    plot_profits(weeks_list, basic_profits, llm_profits, f"{target_model} (Agent)", graph_path)
    
    # Logs
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_filename = f"log_agentic_{safe_model_name}_{timestamp}.csv"
    log_path = os.path.join(LOGS_DIR, log_filename)
    
    with open(log_path, mode='w', newline='') as f:
        if weekly_stats:
            writer = csv.DictWriter(f, fieldnames=weekly_stats[0].keys())
            writer.writeheader()
            writer.writerows(weekly_stats)
            
    # Append to global results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.isfile(SIMULATION_RESULTS_CSV)
    with open(SIMULATION_RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "num_weeks", "basic_profit", "llm_profit", "graph_path"])
        
        writer.writerow([
            timestamp,
            f"{target_model} (Agent)",
            sim_weeks,
            f"{basic_final:.2f}",
            f"{llm_final:.2f}",
            graph_filename
        ])
        
    return {
        "timestamp": timestamp,
        "model": target_model,
        "basic_profit": basic_final,
        "llm_profit": llm_final,
        "weekly_stats": weekly_stats
    }

from src.competitive_run import run_competition as run_legacy_competition

def main():
    parser = argparse.ArgumentParser(description="Competitive LLM Vending Machine Simulation")
    parser.add_argument("--model", type=str, help="Model name to use for the LLMMachine")
    parser.add_argument("--weeks", type=int, help="Number of weeks to simulate")
    parser.add_argument("--json-output", action="store_true", help="Output results as JSON to stdout")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--mode", type=str, choices=["agentic", "legacy"], default="agentic", help="Simulation mode: 'agentic' (LangGraph) or 'legacy' (Standard Loop)")
    
    args = parser.parse_args()
    
    verbose = args.verbose
    if not args.json_output and not args.verbose:
        # Default verbose to True for human usage if not explicitly silenced or json output
        verbose = True
        
    if args.mode == "legacy":
        result = run_legacy_competition(
            model_name=args.model,
            num_weeks=args.weeks,
            verbose=verbose,
            save_plot=True,
            record_history=True
        )
    else:
        result = run_agentic_competition(
            model_name=args.model,
            num_weeks=args.weeks,
            verbose=verbose
        )
    
    if args.json_output:
        print(json.dumps(result))

if __name__ == "__main__":
    main()
