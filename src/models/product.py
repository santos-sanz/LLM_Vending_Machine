class Product:
    def __init__(self, name: str, price: float, cost: float, stock: int, max_stock: int, base_likelihood: float, price_sensitivity: float):
        self.name = name
        self.price = price
        self.cost = cost
        self.stock = stock
        self.max_stock = max_stock
        self.base_likelihood = base_likelihood
        self.price_sensitivity = price_sensitivity

    @property
    def purchase_likelihood(self) -> float:
        # Calculate likelihood: base likelihood minus price * sensitivity
        # Ensure the likelihood is between 0 and 1
        calculated_likelihood = self.base_likelihood - (self.price * self.price_sensitivity)
        return max(0.0, min(1.0, calculated_likelihood))

    def __repr__(self):
        return f"Product(name='{self.name}', price={self.price}, stock={self.stock}/{self.max_stock}, likelihood={self.purchase_likelihood:.2f})"
