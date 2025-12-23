from src.models.vending_machine import VendingMachine

class VendingMachineTools:
    def __init__(self, machines: dict[str, VendingMachine]):
        self.machines = machines

    def change_price(self, machine_name: str, product_name: str, new_price: float):
        """Changes the price of a product in a specific machine."""
        machine = self.machines.get(machine_name)
        if not machine:
            return f"Error: Machine '{machine_name}' not found."
        
        product = machine.products.get(product_name)
        if not product:
            return f"Error: Product '{product_name}' not found in machine '{machine_name}'."
        
        old_price = product.price
        product.price = new_price
        return f"Successfully changed price of '{product_name}' in '{machine_name}' from {old_price} to {new_price}."

    def get_market_data(self):
        """Returns the current state of all vending machines."""
        data = {}
        for name, machine in self.machines.items():
            data[name] = {
                "cash": machine.cash,
                "profit_loss": machine.calculate_profit_loss(),
                "products": {
                    p_name: {
                        "price": p.price,
                        "cost": p.cost,
                        "stock": p.stock,
                        "max_stock": p.max_stock,
                        "likelihood": p.purchase_likelihood
                    } for p_name, p in machine.products.items()
                }
            }
        return data
