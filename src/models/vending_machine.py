from src.models.product import Product

class VendingMachine:
    def __init__(self, initial_cash: float, maintenance_cost: float, initial_investment: float):
        self.products = {}
        self.cash = initial_cash
        self.maintenance_cost = maintenance_cost
        self.initial_investment = initial_investment

    def add_product(self, product: Product):
        if product.name in self.products:
            print(f"Product {product.name} already exists. Updating stock and investment.")
            # If product exists, just update its stock and adjust investment if necessary
            existing_product = self.products[product.name]
            self.initial_investment += (product.stock - existing_product.stock) * product.cost
            self.products[product.name] = product
        else:
            self.products[product.name] = product
            self.initial_investment += product.stock * product.cost
        print(f"Added/Updated product {product.name}. Initial investment now: {self.initial_investment:.2f}")

    def recharge_products(self):
        for product_name, product in self.products.items():
            units_to_recharge = product.max_stock - product.stock
            if units_to_recharge > 0:
                recharge_cost = units_to_recharge * product.cost
                self.cash -= recharge_cost
                product.stock = product.max_stock
                print(f"Recharged {units_to_recharge} units of {product_name}. New cash: {self.cash:.2f}")
            else:
                print(f"Product {product_name} is already at max stock.")

    def sell_product(self, product_name: str) -> bool:
        product = self.products.get(product_name)
        if product and product.stock > 0:
            product.stock -= 1
            self.cash += product.price
            print(f"Sold one {product_name}. Cash: {self.cash:.2f}, Stock left: {product.stock}")
            return True
        elif product and product.stock == 0:
            print(f"{product_name} is out of stock.")
            return False
        else:
            print(f"Product {product_name} not found.")
            return False

    def apply_maintenance(self):
        self.cash -= self.maintenance_cost
        print(f"Applied maintenance cost of {self.maintenance_cost:.2f}. New cash: {self.cash:.2f}")

    def calculate_profit_loss(self) -> float:
        current_value = self.cash + sum(p.stock * p.cost for p in self.products.values())
        profit_loss = self.cash - self.initial_investment
        return profit_loss

    def __repr__(self):
        product_list = '\n  '.join(str(p) for p in self.products.values())
        return (
            f"VendingMachine(\n"
            f"  Cash: {self.cash:.2f},\n"
            f"  Maintenance Cost: {self.maintenance_cost:.2f},\n"
            f"  Initial Investment: {self.initial_investment:.2f},\n"
            f"  Products:\n    {product_list}\n"
            f")"
        )
