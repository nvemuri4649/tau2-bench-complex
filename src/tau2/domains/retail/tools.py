"""Toolkit for the retail domain."""

import functools
import json
import os
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List

from tau2.domains.retail.data_model import (
    GiftCard,
    Order,
    OrderPayment,
    PaymentMethod,
    Product,
    RetailDB,
    User,
    UserAddress,
    Variant,
)
from tau2.domains.retail.utils import RETAIL_DB_PATH
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# ============================================================================
# VERBOSE RESPONSE CONFIGURATION
# ============================================================================
# Environment variable to enable/disable verbose responses
# Set TAU2_VERBOSE_RESPONSES=1 to enable, TAU2_VERBOSE_RESPONSES=0 to disable
VERBOSE_RESPONSES_ENABLED = os.environ.get("TAU2_VERBOSE_RESPONSES", "0") == "1"

# Target: Stay well under 128k context window
# With ~15 tool calls, need ~3k tokens per response = ~45k total
VERBOSE_TARGET_TOKENS_PER_RESPONSE = 12000  # ~3k actual tokens per response
VERBOSE_TARGET_CHARS = VERBOSE_TARGET_TOKENS_PER_RESPONSE * 4


def _generate_retail_audit_log(entity_id: str, entity_type: str, num_entries: int = 300) -> List[Dict]:
    """Generate realistic retail system audit log entries."""
    actions = [
        "ORDER_CREATED", "ORDER_UPDATED", "PAYMENT_PROCESSED", "PAYMENT_FAILED",
        "INVENTORY_CHECKED", "INVENTORY_RESERVED", "INVENTORY_RELEASED", "SHIPPING_LABEL_CREATED",
        "TRACKING_UPDATED", "DELIVERY_ATTEMPTED", "DELIVERY_CONFIRMED", "RETURN_INITIATED",
        "RETURN_RECEIVED", "REFUND_PROCESSED", "EXCHANGE_REQUESTED", "CUSTOMER_CONTACTED",
        "ADDRESS_VALIDATED", "FRAUD_CHECK_PASSED", "PRICE_ADJUSTED", "COUPON_APPLIED",
        "GIFT_WRAP_ADDED", "SHIPPING_UPGRADED", "ITEM_BACKORDERED", "ITEM_CANCELLED"
    ]
    services = ["order-service", "payment-gateway", "inventory-mgmt", "shipping-service",
                "warehouse-mgmt", "customer-service", "fraud-detection", "pricing-engine"]
    entries = []
    base_time = datetime(2025, 2, 25, 10, 0, 0)
    for i in range(num_entries):
        entry_time = base_time - timedelta(hours=i * 2, minutes=random.randint(0, 59))
        entries.append({
            "log_id": str(uuid.uuid4()),
            "timestamp": entry_time.isoformat() + "Z",
            "entity_id": entity_id,
            "entity_type": entity_type,
            "action": random.choice(actions),
            "service": random.choice(services),
            "warehouse_id": f"WH-{random.randint(100, 999)}",
            "worker_id": f"WRK{random.randint(10000, 99999)}",
            "processing_time_ms": random.randint(10, 200),
            "cache_hit": random.choice([True, False]),
            "shipping_zone": random.choice(["A", "B", "C", "D", "E"]),
            "carrier": random.choice(["UPS", "FedEx", "USPS", "DHL", "OnTrac"]),
        })
    return entries


def _generate_similar_orders(actual_order_id: str, actual_user: str) -> List[Dict]:
    """Generate confounding similar order data."""
    statuses = ["pending", "processing", "shipped", "delivered", "cancelled", "returned"]
    
    similar_orders = []
    order_prefix = actual_order_id[:3] if actual_order_id else "#W0"
    
    for i in range(8):
        order_num = random.randint(1000000, 9999999)
        similar_orders.append({
            "order_id": f"{order_prefix}{order_num}",
            "user_id": f"{actual_user[:5]}_{random.randint(100, 999)}" if actual_user else f"user_{random.randint(100, 999)}",
            "status": random.choice(statuses),
            "total": round(random.uniform(20, 500), 2),
            "items_count": random.randint(1, 8),
            "order_date": f"2025-{random.randint(1, 2):02d}-{random.randint(1, 28):02d}",
            "similarity_score": round(random.uniform(0.60, 0.88), 2),
            "match_type": random.choice(["user_match", "address_similar", "item_overlap", "recent_order"]),
            "_note": "Similar order found - verify correct order before proceeding"
        })
    return similar_orders


def _generate_similar_products(actual_product_id: str, actual_name: str) -> List[Dict]:
    """Generate confounding similar product data."""
    categories = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books", "Toys"]
    
    similar_products = []
    for i in range(6):
        similar_products.append({
            "product_id": str(random.randint(1000000000, 9999999999)),
            "name": f"{actual_name[:20] if actual_name else 'Product'}... (variant {i+1})",
            "category": random.choice(categories),
            "price_range": f"${random.randint(10, 100)}-${random.randint(100, 500)}",
            "variants_count": random.randint(2, 12),
            "in_stock": random.choice([True, True, True, False]),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews_count": random.randint(10, 5000),
            "similarity_score": round(random.uniform(0.65, 0.92), 2),
            "_note": "Similar product - verify correct item before processing"
        })
    return similar_products


def _generate_similar_users(actual_user_id: str, actual_name: str) -> List[Dict]:
    """Generate confounding similar user data."""
    first_names = ["Sara", "Sarah", "John", "Jon", "Mike", "Michael", "Lisa", "Liza"]
    last_names = ["Doe", "Smith", "Johnson", "Williams", "Brown", "Jones", "Davis"]
    
    similar_users = []
    for i in range(6):
        similar_users.append({
            "user_id": f"{random.choice(first_names).lower()}_{random.choice(last_names).lower()}_{random.randint(100, 999)}",
            "name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "email_domain": random.choice(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]),
            "zip_code": f"{random.randint(10000, 99999)}",
            "account_status": random.choice(["active", "active", "active", "suspended"]),
            "orders_count": random.randint(1, 50),
            "similarity_score": round(random.uniform(0.55, 0.85), 2),
            "match_type": random.choice(["name_partial", "email_similar", "address_match", "phone_similar"]),
            "_note": "Similar user found - verify identity before proceeding"
        })
    return similar_users


def _generate_inventory_history(product_id: str, num_entries: int = 40) -> List[Dict]:
    """Generate inventory movement history."""
    entries = []
    base_qty = random.randint(50, 500)
    
    for i in range(num_entries):
        qty_change = random.randint(-20, 30)
        entries.append({
            "record_id": str(uuid.uuid4()),
            "product_id": product_id,
            "timestamp": f"2025-02-{max(1, 25-i):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
            "warehouse_id": f"WH-{random.randint(100, 999)}",
            "quantity_before": base_qty,
            "quantity_after": base_qty + qty_change,
            "movement_type": random.choice(["sale", "return", "restock", "transfer", "adjustment"]),
            "reference_id": f"REF-{random.randint(100000, 999999)}",
        })
        base_qty += qty_change
    return entries


def _generate_verbose_padding_retail(entity_id: str, entity_type: str) -> Dict:
    """Generate retail-specific verbose padding with confounding data."""
    padding = {
        "_audit_log": _generate_retail_audit_log(entity_id, entity_type, num_entries=200),
        "_inventory_movements": _generate_inventory_history(entity_id, num_entries=25),
        "_system_diagnostics": {
            "inventory_sync_status": "synchronized",
            "last_sync_time": "2025-02-25T12:05:00Z",
            "cache_status": {"hit_rate": 0.91, "entries": 32045, "ttl_seconds": 180},
            "warehouse_status": {
                "WH-101": {"status": "operational", "capacity": "78%"},
                "WH-205": {"status": "operational", "capacity": "65%"},
                "WH-309": {"status": "maintenance", "capacity": "45%"}
            }
        },
        "_fulfillment_options": {
            "standard_shipping": {"available": True, "days": "5-7", "cost": 5.99},
            "express_shipping": {"available": True, "days": "2-3", "cost": 15.99},
            "next_day": {"available": random.choice([True, False]), "cost": 25.99},
            "store_pickup": {"available": True, "locations": random.randint(2, 8)}
        },
        "_internal_metadata": {
            "response_generated_at": datetime.now().isoformat() + "Z",
            "processing_pipeline": [
                {"stage": "request_validation", "duration_ms": 2, "status": "success"},
                {"stage": "inventory_lookup", "duration_ms": 35, "status": "success"},
                {"stage": "pricing_calculation", "duration_ms": 18, "status": "success"},
                {"stage": "availability_check", "duration_ms": 42, "status": "success"},
                {"stage": "response_formatting", "duration_ms": 5, "status": "success"},
            ],
            "api_version": "v3.2.0"
        }
    }
    return padding


def add_verbose_padding_retail(func):
    """Decorator that adds verbose padding to retail tool responses."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not VERBOSE_RESPONSES_ENABLED:
            return result
            
        # Convert Pydantic models or other objects to dict if needed
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif hasattr(result, '__dict__') and not isinstance(result, dict):
            result = {"data": str(result)}
        elif isinstance(result, list):
            result = {"results": result}
        elif not isinstance(result, dict):
            result = {"value": result}
        
        # Extract entity info
        entity_id = result.get('order_id') or result.get('user_id') or \
                   result.get('product_id') or str(uuid.uuid4())[:8]
        entity_type = result.get('_entity_type', 'order')
        
        # Add verbose padding
        result.update(_generate_verbose_padding_retail(str(entity_id), entity_type))
        
        return result
    
    return wrapper


def add_confounding_orders(func):
    """Decorator that adds confounding similar orders to responses."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not VERBOSE_RESPONSES_ENABLED:
            return result
            
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif not isinstance(result, dict):
            return result
            
        order_id = result.get('order_id', '#W0000000')
        user_id = result.get('user_id', 'unknown_user')
        
        result['_similar_orders_found'] = _generate_similar_orders(order_id, user_id)
        result['_search_notes'] = "Multiple similar orders found in system. Exact match returned based on order ID. Review _similar_orders_found for potential related orders."
        
        return result
    
    return wrapper


def add_confounding_users(func):
    """Decorator that adds confounding similar users to responses."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not VERBOSE_RESPONSES_ENABLED:
            return result
            
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif not isinstance(result, dict):
            return result
            
        user_id = result.get('user_id', 'unknown')
        name = result.get('name', {})
        full_name = f"{name.get('first_name', 'Unknown')} {name.get('last_name', 'User')}" if isinstance(name, dict) else str(name)
        
        result['_similar_users_found'] = _generate_similar_users(user_id, full_name)
        result['_search_notes'] = "Multiple similar user profiles found. Verify correct customer identity before making account changes."
        
        return result
    
    return wrapper


def add_confounding_products(func):
    """Decorator that adds confounding similar products to responses."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if not VERBOSE_RESPONSES_ENABLED:
            return result
            
        if hasattr(result, 'model_dump'):
            result = result.model_dump()
        elif not isinstance(result, dict):
            return result
            
        product_id = result.get('product_id', 'unknown')
        name = result.get('name', 'Product')
        
        result['_similar_products_found'] = _generate_similar_products(product_id, name)
        result['_related_items'] = [
            {"type": "frequently_bought_together", "count": random.randint(3, 8)},
            {"type": "customers_also_viewed", "count": random.randint(5, 15)},
            {"type": "same_category", "count": random.randint(20, 100)}
        ]
        
        return result
    
    return wrapper


class RetailTools(ToolKitBase):  # Tools
    """All the tools for the retail domain."""

    db: RetailDB

    def __init__(self, db: RetailDB) -> None:
        super().__init__(db)

    def _get_order(self, order_id: str) -> Order:
        """Get the order from the database.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.

        Returns:
            The order.

        Raises:
            ValueError: If the order is not found.
        """
        if order_id not in self.db.orders:
            raise ValueError("Order not found")
        return self.db.orders[order_id]

    def _get_user(self, user_id: str) -> User:
        """Get the user from the database.

        Args:
            user_id: The user id, such as 'sara_doe_496'.

        Returns:
            The user.

        Raises:
            ValueError: If the user is not found.
        """
        if user_id not in self.db.users:
            raise ValueError("User not found")
        return self.db.users[user_id]

    def _get_product(self, product_id: str) -> Product:
        """Get the product from the database.

        Args:
            product_id: The product id, such as '6086499569'. Be careful the product id is different from the item id.

        Returns:
            The product.

        Raises:
            ValueError: If the product is not found.
        """
        if product_id not in self.db.products:
            raise ValueError("Product not found")
        return self.db.products[product_id]

    def _get_variant(self, product_id: str, variant_id: str) -> Variant:
        """Get the variant from the database.

        Args:
            product_id: The product id, such as '6086499569'. Be careful the product id is different from the item id.
            variant_id: The variant id, such as '1008292230'.

        Returns:
            The variant.

        Raises:
            ValueError: If the variant is not found.
        """
        product = self._get_product(product_id)
        if variant_id not in product.variants:
            raise ValueError("Variant not found")
        return product.variants[variant_id]

    def _get_payment_method(
        self, user_id: str, payment_method_id: str
    ) -> PaymentMethod:
        """Get the payment method from the database.

        Args:
            payment_method_id: The payment method id, such as 'gift_card_0000000' or 'credit_card_0000000'.

        Returns:
            The payment method.

        Raises:
            ValueError: If the payment method is not found.
        """
        user = self._get_user(user_id)
        if payment_method_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        return user.payment_methods[payment_method_id]

    def _is_pending_order(self, order: Order) -> bool:
        """Check if the order is pending. This is not a strict check, and not meant to be used for modify_items in pending orders.

        Args:
            order: The order.
        """
        return "pending" in order.status

    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.

        Returns:
            The result of the mathematical expression.

        Raises:
            ValueError: If the expression is invalid.
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.WRITE)
    def cancel_pending_order(self, order_id: str, reason: str) -> Order:
        """Cancel a pending order. If the order is already processed or delivered,
        it cannot be cancelled. The agent needs to explain the cancellation detail
        and ask for explicit user confirmation (yes/no) to proceed. If the user confirms,
        the order status will be changed to 'cancelled' and the payment will be refunded.
        The refund will be added to the user's gift card balance immediately if the payment
        was made using a gift card, otherwise the refund would take 5-7 business days to process.
        The function returns the order details after the cancellation.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            reason: The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.

        Returns:
            Order: The order details after the cancellation.
        """
        # check order exists and is pending
        order = self._get_order(order_id)
        if order.status != "pending":
            raise ValueError("Non-pending order cannot be cancelled")

        # check reason
        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason")

        # handle refund
        refunds = []
        for payment in order.payment_history:
            payment_id = payment.payment_method_id
            refund = OrderPayment(
                transaction_type="refund",
                amount=payment.amount,
                payment_method_id=payment_id,
            )
            refunds.append(refund)
            user = self._get_user(order.user_id)
            payment_method = self._get_payment_method(user.user_id, payment_id)
            if isinstance(payment_method, GiftCard):  # refund to gift card immediately
                payment_method.balance += payment.amount
                payment_method.balance = round(payment_method.balance, 2)

        # update order status
        order.status = "cancelled"
        order.cancel_reason = reason
        order.payment_history.extend(refunds)

        return order

    @is_tool(ToolType.WRITE)
    def exchange_delivered_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str,
    ) -> Order:
        """Exchange items in a delivered order to new items of the same product type.
        For a delivered order, return or exchange can be only done once by the agent.
        The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            item_ids: The item ids to be exchanged, each such as '1008292230'. There could be duplicate items in the list.
            new_item_ids: The item ids to be exchanged for, each such as '1008292230'.
                         There could be duplicate items in the list. Each new item id should match the item id
                         in the same position and be of the same product.
            payment_method_id: The payment method id to pay or receive refund for the item price difference,
                             such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up
                             from the user or order details.

        Returns:
            Order: The order details after the exchange.

        Raises:
            ValueError: If the order is not delivered.
            ValueError: If the items to be exchanged do not exist.
            ValueError: If the new items do not exist or do not match the old items.
            ValueError: If the number of items to be exchanged does not match.
        """
        # check order exists and is delivered
        order = self._get_order(order_id)
        if order.status != "delivered":
            raise ValueError("Non-delivered order cannot be exchanged")

        # check the items to be exchanged exist. There can be duplicate items in the list.
        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"Number of {item_id} not found.")

        # check new items exist and match old items and are available
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match.")

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = next((item for item in order.items if item.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            product_id = item.product_id
            variant = self._get_variant(product_id, new_item_id)
            if not variant.available:
                raise ValueError(f"New item {new_item_id} not found or available")

            old_price = item.price
            new_price = variant.price
            diff_price += new_price - old_price

        diff_price = round(diff_price, 2)

        # check payment method exists and can cover the price difference if gift card
        payment_method = self._get_payment_method(order.user_id, payment_method_id)

        if isinstance(payment_method, GiftCard) and payment_method.balance < diff_price:
            raise ValueError(
                "Insufficient gift card balance to pay for the price difference"
            )

        # modify the order
        order.status = "exchange requested"
        order.exchange_items = sorted(item_ids)
        order.exchange_new_items = sorted(new_item_ids)
        order.exchange_payment_method_id = payment_method_id
        order.exchange_price_difference = diff_price

        return order

    @is_tool(ToolType.READ)
    def find_user_id_by_name_zip(
        self, first_name: str, last_name: str, zip: str
    ) -> str:
        """Find user id by first name, last name, and zip code. If the user is not found, the function
        will return an error message. By default, find user id by email, and only call this function
        if the user is not found by email or cannot remember email.

        Args:
            first_name: The first name of the customer, such as 'John'.
            last_name: The last name of the customer, such as 'Doe'.
            zip: The zip code of the customer, such as '12345'.

        Returns:
            str: The user id if found, otherwise an error message.

        Raises:
            ValueError: If the user is not found.
        """
        for user_id, user in self.db.users.items():
            if (
                user.name.first_name.lower() == first_name.lower()
                and user.name.last_name.lower() == last_name.lower()
                and user.address.zip == zip
            ):
                return user_id
        raise ValueError("User not found")

    @is_tool(ToolType.READ)
    def find_user_id_by_email(self, email: str) -> str:
        """Find user id by email. If the user is not found, the function will return an error message.

        Args:
            email: The email of the user, such as 'something@example.com'.

        Returns:
            str: The user id if found, otherwise an error message.

        Raises:
            ValueError: If the user is not found.
        """
        for user_id, user in self.db.users.items():
            if user.email.lower() == email.lower():
                return user_id
        raise ValueError("User not found")

    @is_tool(ToolType.READ)
    @add_verbose_padding_retail
    @add_confounding_orders
    def get_order_details(self, order_id: str):
        """Get the status and details of an order.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.

        Returns:
            Order: The order details.

        Raises:
            ValueError: If the order is not found.
        """
        order = self._get_order(order_id)
        return order.model_dump() if hasattr(order, 'model_dump') else order

    @is_tool(ToolType.READ)
    @add_verbose_padding_retail
    @add_confounding_products
    def get_product_details(self, product_id: str):
        """Get the inventory details of a product.

        Args:
            product_id: The product id, such as '6086499569'. Be careful the product id is different from the item id.

        Returns:
            Product: The product details.

        Raises:
            ValueError: If the product is not found.
        """
        product = self._get_product(product_id)
        return product.model_dump() if hasattr(product, 'model_dump') else product

    @is_tool(ToolType.READ)
    @add_verbose_padding_retail
    @add_confounding_users
    def get_user_details(self, user_id: str):
        """Get the details of a user, including their orders.

        Args:
            user_id: The user id, such as 'sara_doe_496'.

        Returns:
            User: The user details.

        Raises:
            ValueError: If the user is not found.
        """
        user = self._get_user(user_id)
        return user.model_dump() if hasattr(user, 'model_dump') else user

    @is_tool(ToolType.READ)
    def list_all_product_types(self) -> str:
        """List the name and product id of all product types.
        Each product type has a variety of different items with unique item ids and options.
        There are only 50 product types in the store.

        Returns:
            str: A JSON string mapping product names to their product IDs, sorted alphabetically by name.
        """
        product_dict = {
            product.name: product.product_id for product in self.db.products.values()
        }
        return json.dumps(product_dict, sort_keys=True)

    @is_tool(ToolType.WRITE)
    def modify_pending_order_address(
        self,
        order_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> Order:
        """Modify the shipping address of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            address1: The first line of the address, such as '123 Main St'.
            address2: The second line of the address, such as 'Apt 1' or ''.
            city: The city, such as 'San Francisco'.
            state: The state, such as 'CA'.
            country: The country, such as 'USA'.
            zip: The zip code, such as '12345'.

        Returns:
            Order: The order details after the modification.

        Raises:
            ValueError: If the order is not pending.
        """
        # Check if the order exists and is pending
        order = self._get_order(order_id)
        if not self._is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")

        # Modify the address
        order.address = UserAddress(
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )
        return order

    @is_tool(ToolType.WRITE)
    def modify_pending_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        new_item_ids: List[str],
        payment_method_id: str,
    ) -> Order:
        """Modify items in a pending order to new items of the same product type. For a pending order, this function can only be called once. The agent needs to explain the exchange detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            item_ids: The item ids to be modified, each such as '1008292230'. There could be duplicate items in the list.
            new_item_ids: The item ids to be modified for, each such as '1008292230'. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.
            payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.

        Returns:
            Order: The order details after the modification.

        Raises:
            ValueError: If the order is not pending.
            ValueError: If the items to be modified do not exist.
            ValueError: If the new items do not exist or do not match the old items.
            ValueError: If the number of items to be modified does not match.
        """

        # Check if the order exists and is pending
        order = self._get_order(order_id)
        if order.status != "pending":
            raise ValueError("Non-pending order cannot be modified")

        # Check if the items to be modified exist. There can be duplicate items in the list.
        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"{item_id} not found")

        # Check new items exist, match old items, and are available
        if len(item_ids) != len(new_item_ids):
            raise ValueError("The number of items to be exchanged should match")

        diff_price = 0
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            if item_id == new_item_id:
                raise ValueError(
                    "The new item id should be different from the old item id"
                )
            item = next((item for item in order.items if item.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            product_id = item.product_id
            variant = self._get_variant(product_id, new_item_id)
            if not variant.available:
                raise ValueError(f"New item {new_item_id} not found or available")

            old_price = item.price
            new_price = variant.price
            diff_price += new_price - old_price

        # Check if the payment method exists
        payment_method = self._get_payment_method(order.user_id, payment_method_id)

        # If the new item is more expensive, check if the gift card has enough balance
        if isinstance(payment_method, GiftCard) and payment_method.balance < diff_price:
            raise ValueError("Insufficient gift card balance to pay for the new item")

        # Handle the payment or refund
        order.payment_history.append(
            OrderPayment(
                transaction_type="payment" if diff_price > 0 else "refund",
                amount=abs(diff_price),
                payment_method_id=payment_method_id,
            )
        )
        if isinstance(payment_method, GiftCard):
            payment_method.balance -= diff_price
            payment_method.balance = round(payment_method.balance, 2)

        # Modify the order
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = next((item for item in order.items if item.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            item.item_id = new_item_id
            item.price = variant.price
            item.options = variant.options
        order.status = "pending (item modified)"

        return order

    @is_tool(ToolType.WRITE)
    def modify_pending_order_payment(
        self,
        order_id: str,
        payment_method_id: str,
    ) -> Order:
        """Modify the payment method of a pending order. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details.

        Returns:
            Order: The order details after the modification.

        Raises:
            ValueError: If the order is not pending.
            ValueError: If the payment method does not exist.
            ValueError: If the payment history has more than one payment.
            ValueError: If the new payment method is the same as the current one.
        """
        order = self._get_order(order_id)

        # Check if the order exists and is pending
        if not self._is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")

        # Check if the payment method exists
        payment_method = self._get_payment_method(order.user_id, payment_method_id)

        # Check that the payment history should only have one payment
        if (
            len(order.payment_history) != 1
            or order.payment_history[0].transaction_type != "payment"
        ):
            raise ValueError("There should be exactly one payment for a pending order")

        # Check that the payment method is different
        if order.payment_history[0].payment_method_id == payment_method_id:
            raise ValueError(
                "The new payment method should be different from the current one"
            )

        amount = order.payment_history[0].amount

        # Check if the new payment method has enough balance if it is a gift card
        if isinstance(payment_method, GiftCard) and payment_method.balance < amount:
            raise ValueError("Insufficient gift card balance to pay for the order")

        # Modify the payment method
        order.payment_history.extend(
            [
                OrderPayment(
                    transaction_type="payment",
                    amount=amount,
                    payment_method_id=payment_method_id,
                ),
                OrderPayment(
                    transaction_type="refund",
                    amount=amount,
                    payment_method_id=order.payment_history[0].payment_method_id,
                ),
            ]
        )

        # If payment is made by gift card, update the balance
        if isinstance(payment_method, GiftCard):
            payment_method.balance -= amount
            payment_method.balance = round(payment_method.balance, 2)

        # If refund is made to a gift card, update the balance
        old_payment_method = self._get_payment_method(
            order.user_id, order.payment_history[0].payment_method_id
        )
        if isinstance(old_payment_method, GiftCard):
            old_payment_method.balance += amount
            old_payment_method.balance = round(old_payment_method.balance, 2)

        return order

    @is_tool(ToolType.WRITE)
    def modify_user_address(
        self,
        user_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> User:
        """Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.

        Args:
            user_id: The user id, such as 'sara_doe_496'.
            address1: The first line of the address, such as '123 Main St'.
            address2: The second line of the address, such as 'Apt 1' or ''.
            city: The city, such as 'San Francisco'.
            state: The state, such as 'CA'.
            country: The country, such as 'USA'.
            zip: The zip code, such as '12345'.

        Returns:
            User: The user details after the modification.

        Raises:
            ValueError: If the user is not found.
        """
        user = self._get_user(user_id)
        user.address = UserAddress(
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )
        return user

    @is_tool(ToolType.WRITE)
    def return_delivered_order_items(
        self,
        order_id: str,
        item_ids: List[str],
        payment_method_id: str,
    ) -> Order:
        """Return some items of a delivered order.
        The order status will be changed to 'return requested'.
        The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed.
        The user will receive follow-up email for how and where to return the item.

        Args:
            order_id: The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.
            item_ids: The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.
            payment_method_id: The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'.
                             These can be looked up from the user or order details.

        Returns:
            Order: The order details after requesting the return.

        Raises:
            ValueError: If the order is not delivered.
            ValueError: If the payment method is not the original payment method or a gift card.
            ValueError: If the items to be returned do not exist.
        """
        order = self._get_order(order_id)
        if order.status != "delivered":
            raise ValueError("Non-delivered order cannot be returned")

        # Check if the payment method exists and is either the original payment method or a gift card
        user = self._get_user(order.user_id)
        payment_method = self._get_payment_method(user.user_id, payment_method_id)

        if (
            not isinstance(payment_method, GiftCard)
            and payment_method_id != order.payment_history[0].payment_method_id
        ):
            raise ValueError("Payment method should be the original payment method")

        # Check if the items to be returned exist (there could be duplicate items in either list)
        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids:
            if item_ids.count(item_id) > all_item_ids.count(item_id):
                raise ValueError("Some item not found")

        # Update the order status
        order.status = "return requested"
        order.return_items = sorted(item_ids)
        order.return_payment_method_id = payment_method_id

        return order

    # @is_tool(ToolType.THINK)
    # def think(self, thought: str) -> str:
    #     """
    #     Use the tool to think about something.
    #     It will not obtain new information or change the database, but just append the thought to the log.
    #     Use it when complex reasoning or some cache memory is needed.

    #     Args:
    #         thought: A thought to think about.

    #     Returns:
    #         Empty string
    #     """
    #     return ""

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    # ==================== ASSERTION FUNCTIONS FOR EVALUATION ====================
    # These functions are used by the evaluation system to verify task completion

    def assert_order_status(self, order_id: str, expected_status: str) -> bool:
        """Assert that an order has the expected status."""
        order = self._get_order(order_id)
        return order.status == expected_status

    def assert_order_item_count(self, order_id: str, expected_count: int) -> bool:
        """Assert that an order has the expected number of items."""
        order = self._get_order(order_id)
        return len(order.items) == expected_count

    def assert_order_has_item(self, order_id: str, item_id: str) -> bool:
        """Assert that an order contains a specific item."""
        order = self._get_order(order_id)
        return any(item.item_id == item_id for item in order.items)

    def assert_order_cancel_reason(self, order_id: str, expected_reason: str) -> bool:
        """Assert that a cancelled order has the expected reason."""
        order = self._get_order(order_id)
        return order.cancel_reason == expected_reason

    def assert_order_exchange_items(self, order_id: str, expected_item_ids: list) -> bool:
        """Assert that an order has the expected exchange items."""
        order = self._get_order(order_id)
        if order.exchange_items is None:
            return expected_item_ids is None or len(expected_item_ids) == 0
        return sorted(order.exchange_items) == sorted(expected_item_ids)

    def assert_order_return_items(self, order_id: str, expected_item_ids: list) -> bool:
        """Assert that an order has the expected return items."""
        order = self._get_order(order_id)
        if order.return_items is None:
            return expected_item_ids is None or len(expected_item_ids) == 0
        return sorted(order.return_items) == sorted(expected_item_ids)

    def assert_user_gift_card_balance(self, user_id: str, gift_card_id: str, expected_balance: float) -> bool:
        """Assert that a user's gift card has the expected balance."""
        user = self._get_user(user_id)
        if gift_card_id not in user.payment_methods:
            return False
        pm = user.payment_methods[gift_card_id]
        return isinstance(pm, GiftCard) and abs(pm.balance - expected_balance) < 0.01

    def assert_user_order_count(self, user_id: str, expected_count: int) -> bool:
        """Assert that a user has the expected number of orders."""
        user = self._get_user(user_id)
        return len(user.orders) == expected_count

    def assert_order_address_zip(self, order_id: str, expected_zip: str) -> bool:
        """Assert that an order has the expected shipping zip code."""
        order = self._get_order(order_id)
        return order.address.zip == expected_zip

    def assert_order_address_city(self, order_id: str, expected_city: str) -> bool:
        """Assert that an order has the expected shipping city."""
        order = self._get_order(order_id)
        return order.address.city == expected_city

    def assert_product_variant_available(self, product_id: str, variant_id: str) -> bool:
        """Assert that a product variant is available."""
        variant = self._get_variant(product_id, variant_id)
        return variant.available

    def assert_order_payment_method(self, order_id: str, expected_payment_method_id: str) -> bool:
        """Assert that an order's first payment was made with the expected method."""
        order = self._get_order(order_id)
        if not order.payment_history:
            return False
        return order.payment_history[0].payment_method_id == expected_payment_method_id

    def assert_order_total_payment(self, order_id: str, expected_total: float) -> bool:
        """Assert that an order's total payments match expected amount."""
        order = self._get_order(order_id)
        total = sum(p.amount for p in order.payment_history if p.transaction_type == "payment")
        return abs(total - expected_total) < 0.01

    def assert_user_address_zip(self, user_id: str, expected_zip: str) -> bool:
        """Assert that a user's default address has the expected zip code."""
        user = self._get_user(user_id)
        return user.address.zip == expected_zip



if __name__ == "__main__":
    from tau2.domains.retail.utils import RETAIL_DB_PATH

    retail = RetailTools(RetailDB.load(RETAIL_DB_PATH))
    print(retail.get_statistics())
