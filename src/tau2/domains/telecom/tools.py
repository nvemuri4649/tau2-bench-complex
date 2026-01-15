"""Toolkit for the telecom system."""

import uuid
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from tau2.domains.telecom.data_model import (
    Bill,
    BillStatus,
    Customer,
    Device,
    Line,
    LineItem,
    LineStatus,
    Plan,
    TelecomDB,
)
from tau2.domains.telecom.utils import get_today
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

# TODO: Add an abstract base class for the tools


class IDGenerator:
    def __init__(self) -> None:
        self.id_counter = defaultdict(int)

    def get_id(self, id_type: str, id_name: Optional[str] = None) -> str:
        self.id_counter[id_type] += 1
        id_name = id_name or id_type
        return f"{id_name}_{self.id_counter[id_type]}"


class TelecomTools(ToolKitBase):
    """Tools for the telecom domain implementing the functions described in the PRD."""

    db: TelecomDB

    def __init__(self, db: TelecomDB) -> None:
        """Initialize the telecom tools with a database instance."""
        super().__init__(db)
        self.id_generator = IDGenerator()

    # Customer Lookup
    @is_tool(ToolType.READ)
    def get_customer_by_phone(self, phone_number: str) -> Dict[str, Any]:
        """
        Finds a customer by their primary contact or line phone number.
        Returns comprehensive customer information including account metadata,
        line summary, billing summary, and system audit trail.

        Args:
            phone_number: The phone number to search for.

        Returns:
            Dictionary with full customer data including metadata and summaries.
        """
        # Check primary contact number
        found_customer = None
        lookup_method = None
        matched_line_id = None
        
        for customer in self.db.customers:
            if customer.phone_number == phone_number:
                found_customer = customer
                lookup_method = "primary_contact"
                break

            # Check lines
            for line_id in customer.line_ids:
                line = self._get_line_by_id(line_id)
                if line and line.phone_number == phone_number:
                    found_customer = customer
                    lookup_method = "line_association"
                    matched_line_id = line_id
                    break
            if found_customer:
                break

        if not found_customer:
            raise ValueError(f"Customer with phone number {phone_number} not found")
        
        # Build comprehensive response with additional metadata
        lines_summary = []
        total_data_used = 0.0
        active_lines = 0
        suspended_lines = 0
        
        for line_id in found_customer.line_ids:
            try:
                line = self._get_line_by_id(line_id)
                plan = self._get_plan_by_id(line.plan_id)
                device = self._get_device_by_id(line.device_id)
                total_data_used += line.data_used_gb
                if line.status == LineStatus.ACTIVE:
                    active_lines += 1
                elif line.status == LineStatus.SUSPENDED:
                    suspended_lines += 1
                    
                lines_summary.append({
                    "line_id": line.line_id,
                    "phone_number": line.phone_number,
                    "status": line.status.value,
                    "plan_name": plan.name,
                    "device_model": device.model,
                    "data_used_gb": line.data_used_gb,
                    "roaming_enabled": line.roaming_enabled,
                })
            except ValueError:
                lines_summary.append({"line_id": line_id, "error": "Could not retrieve details"})
        
        bills_summary = []
        total_outstanding = 0.0
        for bill_id in found_customer.bill_ids:
            try:
                bill = self._get_bill_by_id(bill_id)
                bills_summary.append({
                    "bill_id": bill.bill_id,
                    "total_due": bill.total_due,
                    "status": bill.status.value,
                    "due_date": str(bill.due_date),
                })
                if bill.status in [BillStatus.OVERDUE, BillStatus.ISSUED]:
                    total_outstanding += bill.total_due
            except ValueError:
                bills_summary.append({"bill_id": bill_id, "error": "Could not retrieve details"})
        
        return {
            "customer_id": found_customer.customer_id,
            "full_name": found_customer.full_name,
            "date_of_birth": found_customer.date_of_birth,
            "email": found_customer.email,
            "phone_number": found_customer.phone_number,
            "account_status": found_customer.account_status.value,
            "address": {
                "street": found_customer.address.street,
                "city": found_customer.address.city,
                "state": found_customer.address.state,
                "zip_code": found_customer.address.zip_code,
            } if found_customer.address else None,
            "payment_methods": [
                {
                    "method_type": pm.method_type,
                    "last_4": pm.account_number_last_4,
                    "expiration": pm.expiration_date,
                }
                for pm in (found_customer.payment_methods or [])
            ],
            "line_ids": found_customer.line_ids,
            "bill_ids": found_customer.bill_ids,
            "created_at": str(found_customer.created_at) if found_customer.created_at else None,
            "goodwill_credit_used_this_year": found_customer.goodwill_credit_used_this_year,
            "_metadata": {
                "lookup_method": lookup_method,
                "matched_line_id": matched_line_id,
                "query_phone_number": phone_number,
                "timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
            },
            "_lines_summary": lines_summary,
            "_billing_summary": {
                "bills": bills_summary,
                "total_outstanding": total_outstanding,
            },
            "_account_metrics": {
                "total_lines": len(found_customer.line_ids),
                "active_lines": active_lines,
                "suspended_lines": suspended_lines,
                "total_data_used_gb": round(total_data_used, 2),
                "account_age_days": (get_today() - found_customer.created_at.date()).days if found_customer.created_at else None,
            },
        }

    @is_tool(ToolType.READ)
    def get_customer_by_id(self, customer_id: str) -> Customer:
        """
        Retrieves a customer directly by their unique ID.

        Args:
            customer_id: The unique identifier of the customer.

        Returns:
            Customer object if found, None otherwise.
        """
        for customer in self.db.customers:
            if customer.customer_id == customer_id:
                return customer

        raise ValueError(f"Customer with ID {customer_id} not found")

    @is_tool(ToolType.READ)
    def get_customer_by_name(self, full_name: str, dob: str) -> List[Customer]:
        """
        Searches for customers by name and DOB. May return multiple matches if names are similar,
        DOB helps disambiguate.

        Args:
            full_name: The full name of the customer.
            dob: Date of birth for verification, in the format YYYY-MM-DD.

        Returns:
            List of matching Customer objects.
        """
        matching_customers = []

        for customer in self.db.customers:
            if (
                customer.full_name.lower() == full_name.lower()
                and customer.date_of_birth == dob
            ):
                matching_customers.append(customer)

        return matching_customers

    # Helper method to get a line by phone number
    def _get_line_by_phone(self, phone_number: str) -> Line:
        """
        Retrieves a line directly by its phone number.

        Args:
            phone_number: The phone number to search for.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified phone number is not found.
        """
        for line in self.db.lines:
            if line.phone_number == phone_number:
                return line
        raise ValueError(f"Line with phone number {phone_number} not found")

    # Helper method to get a line by ID
    def _get_line_by_id(self, line_id: str) -> Line:
        """
        Retrieves a line directly by its unique ID.

        Args:
            line_id: The unique identifier of the line.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified ID is not found.
        """
        for line in self.db.lines:
            if line.line_id == line_id:
                return line
        raise ValueError(f"Line with ID {line_id} not found")

    # Helper method to get a plan by ID
    def _get_plan_by_id(self, plan_id: str) -> Plan:
        """
        Retrieves a plan directly by its unique ID.

        Args:
            plan_id: The unique identifier of the plan.

        Returns:
            Plan object if found.

        Raises:
            ValueError: If the plan with the specified ID is not found.
        """
        for plan in self.db.plans:
            if plan.plan_id == plan_id:
                return plan
        raise ValueError(f"Plan with ID {plan_id} not found")

    # Helper method to get a device by ID
    def _get_device_by_id(self, device_id: str) -> Device:
        """
        Retrieves a device directly by its unique ID.

        Args:
            device_id: The unique identifier of the device.

        Returns:
            Device object if found.

        Raises:
            ValueError: If the device with the specified ID is not found.
        """
        for device in self.db.devices:
            if device.device_id == device_id:
                return device
        raise ValueError(f"Device with ID {device_id} not found")

    # Helper method to get a bill by ID
    def _get_bill_by_id(self, bill_id: str) -> Bill:
        """
        Retrieves a bill directly by its unique ID.

        Args:
            bill_id: The unique identifier of the bill.

        Returns:
            Bill object if found.

        Raises:
            ValueError: If the bill with the specified ID is not found.
        """
        for bill in self.db.bills:
            if bill.bill_id == bill_id:
                return bill
        raise ValueError(f"Bill with ID {bill_id} not found")

    def _get_target_line(self, customer_id: str, line_id: str) -> Line:
        """
        Retrieves a line using the customer ID and line ID.

        Args:
            customer_id: The unique identifier of the customer.
            line_id: The unique identifier of the line.

        Returns:
            Line object if found.

        Raises:
            ValueError: If the line with the specified ID is not found.
        """
        customer = self.get_customer_by_id(customer_id)
        if line_id not in customer.line_ids:
            raise ValueError(f"Line {line_id} not found for customer {customer_id}")
        return self._get_line_by_id(line_id)

    def get_available_plan_ids(self) -> List[str]:
        """
        Returns all the plans that are available to the user.
        """
        return [plan.plan_id for plan in self.db.plans]

    @is_tool(ToolType.READ)
    def get_available_plans(self) -> Dict[str, Any]:
        """
        Retrieves all available mobile plans with comprehensive details including
        pricing, features, comparison metrics, and recommendations for different use cases.

        Returns:
            Dictionary containing full plan catalog with analysis and recommendations.
        """
        plans = []
        for plan in self.db.plans:
            # Calculate plan tier
            if plan.price_per_month <= 20:
                tier = "Entry"
                recommended_for = ["Light users", "Secondary devices", "Kids"]
            elif plan.price_per_month <= 50:
                tier = "Standard"
                recommended_for = ["Average users", "Daily commuters", "Students"]
            elif plan.price_per_month <= 80:
                tier = "Premium"
                recommended_for = ["Heavy users", "Remote workers", "Content streamers"]
            else:
                tier = "Enterprise"
                recommended_for = ["Power users", "Business professionals", "Families"]
            
            plans.append({
                "plan_id": plan.plan_id,
                "name": plan.name,
                "data_limit_gb": plan.data_limit_gb,
                "price_per_month": plan.price_per_month,
                "data_refueling_price_per_gb": plan.data_refueling_price_per_gb,
                "tier": tier,
                "annual_cost": plan.price_per_month * 12,
                "cost_per_gb": round(plan.price_per_month / plan.data_limit_gb, 4) if plan.data_limit_gb > 0 else 0,
                "features": {
                    "international_roaming": plan.data_limit_gb >= 15,
                    "mobile_hotspot": plan.data_limit_gb >= 5,
                    "hd_streaming": plan.data_limit_gb >= 15,
                    "unlimited_talk_text": True,
                    "5g_access": plan.price_per_month >= 40,
                    "wifi_calling": True,
                },
                "recommended_for": recommended_for,
                "data_value_score": round(plan.data_limit_gb / plan.price_per_month * 10, 2) if plan.price_per_month > 0 else 0,
            })
        
        # Sort by price for comparison
        plans_by_price = sorted(plans, key=lambda p: p["price_per_month"])
        
        return {
            "plans": plans,
            "total_plans_available": len(plans),
            "_plan_comparison": {
                "lowest_price_plan": plans_by_price[0]["name"] if plans_by_price else None,
                "highest_data_plan": max(plans, key=lambda p: p["data_limit_gb"])["name"] if plans else None,
                "best_value_plan": max(plans, key=lambda p: p["data_value_score"])["name"] if plans else None,
                "price_range": f"${plans_by_price[0]['price_per_month']:.2f} - ${plans_by_price[-1]['price_per_month']:.2f}" if plans_by_price else None,
            },
            "_recommendations": {
                "for_light_users": next((p["name"] for p in plans if p["tier"] == "Entry"), None),
                "for_heavy_users": next((p["name"] for p in plans if p["tier"] in ["Premium", "Enterprise"]), None),
                "for_families": next((p["name"] for p in plans if "Family" in p["name"]), None),
                "best_refueling_rate": min(plans, key=lambda p: p["data_refueling_price_per_gb"])["name"] if plans else None,
            },
            "_metadata": {
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
                "catalog_version": "2025Q1",
                "prices_valid_until": "2025-12-31",
            },
        }

    @is_tool(ToolType.WRITE)
    def change_plan(
        self, customer_id: str, line_id: str, new_plan_id: str
    ) -> Dict[str, Any]:
        """
        Changes a line's mobile plan. The change takes effect at the start of the
        next billing cycle. Validates the plan exists and customer owns the line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to change the plan for.
            new_plan_id: ID of the new plan to switch to.

        Returns:
            Dictionary with success message and plan details.

        Raises:
            ValueError: If customer, line, or plan not found.
        """
        target_line = self._get_target_line(customer_id, line_id)
        new_plan = self._get_plan_by_id(new_plan_id)
        old_plan = self._get_plan_by_id(target_line.plan_id)
        
        if target_line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to change plan")
        
        target_line.plan_id = new_plan_id
        target_line.last_plan_change_date = get_today()
        
        return {
            "message": f"Plan changed successfully for line {line_id}",
            "old_plan": old_plan.name,
            "new_plan": new_plan.name,
            "old_price": old_plan.price_per_month,
            "new_price": new_plan.price_per_month,
            "effective_date": "Next billing cycle",
        }

    @is_tool(ToolType.WRITE)
    def make_payment(self, customer_id: str, bill_id: str) -> Dict[str, Any]:
        """
        Processes a payment for a bill that is awaiting payment.
        The bill must be in AWAITING_PAYMENT status (use send_payment_request first).
        
        Args:
            customer_id: ID of the customer making the payment.
            bill_id: ID of the bill to pay.
            
        Returns:
            Dictionary with payment confirmation details.
            
        Raises:
            ValueError: If customer not found, bill not found, or bill not awaiting payment.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
            
        if bill_id not in customer.bill_ids:
            raise ValueError(f"Bill {bill_id} not found for customer {customer_id}")
            
        bill = self._get_bill_by_id(bill_id)
        if bill.status != BillStatus.AWAITING_PAYMENT:
            raise ValueError(f"Bill {bill_id} is not awaiting payment. Current status: {bill.status.value}")
        
        bill.status = BillStatus.PAID
        
        return {
            "message": f"Payment of ${bill.total_due:.2f} processed successfully",
            "bill_id": bill_id,
            "amount_paid": bill.total_due,
            "new_status": "Paid",
            "payment_date": str(get_today()),
        }

    @is_tool(ToolType.READ)
    def get_details_by_id(self, id: str) -> Dict[str, Any]:
        """
        Retrieves comprehensive details for a given ID including related entities,
        metadata, usage statistics, and system audit information.
        The ID must be a valid ID for a Customer, Line, Device, Bill, or Plan.

        Args:
            id: The ID of the object to retrieve.

        Returns:
            Comprehensive dictionary with full entity details and related information.

        Raises:
            ValueError: If the ID is not found or if the ID format is invalid.
        """
        if id.startswith("L"):
            line = self._get_line_by_id(id)
            plan = self._get_plan_by_id(line.plan_id)
            device = self._get_device_by_id(line.device_id)
            
            # Calculate data remaining
            data_limit = plan.data_limit_gb
            data_used = line.data_used_gb
            data_remaining = max(0, data_limit - data_used + line.data_refueling_gb)
            data_utilization_pct = (data_used / data_limit * 100) if data_limit > 0 else 0
            
            return {
                "entity_type": "Line",
                "line_id": line.line_id,
                "phone_number": line.phone_number,
                "status": line.status.value,
                "plan_id": line.plan_id,
                "plan_name": plan.name,
                "plan_price_per_month": plan.price_per_month,
                "plan_data_limit_gb": plan.data_limit_gb,
                "plan_refueling_rate_per_gb": plan.data_refueling_price_per_gb,
                "device_id": line.device_id,
                "device_model": device.model,
                "device_type": device.device_type,
                "device_is_esim_capable": device.is_esim_capable,
                "data_used_gb": line.data_used_gb,
                "data_refueling_gb": line.data_refueling_gb,
                "data_remaining_gb": round(data_remaining, 2),
                "roaming_enabled": line.roaming_enabled,
                "contract_end_date": str(line.contract_end_date) if line.contract_end_date else None,
                "last_plan_change_date": str(line.last_plan_change_date) if line.last_plan_change_date else None,
                "last_sim_replacement_date": str(line.last_sim_replacement_date) if line.last_sim_replacement_date else None,
                "suspension_start_date": str(line.suspension_start_date) if line.suspension_start_date else None,
                "_usage_metrics": {
                    "data_utilization_percentage": round(data_utilization_pct, 1),
                    "data_overage_gb": max(0, data_used - data_limit),
                    "estimated_overage_charge": max(0, (data_used - data_limit) * plan.data_refueling_price_per_gb),
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                    "cache_ttl_seconds": 300,
                },
            }
        elif id.startswith("D"):
            device = self._get_device_by_id(id)
            
            return {
                "entity_type": "Device",
                "device_id": device.device_id,
                "device_type": device.device_type,
                "model": device.model,
                "imei": device.imei,
                "is_esim_capable": device.is_esim_capable,
                "activated": device.activated,
                "activation_date": str(device.activation_date) if device.activation_date else None,
                "last_esim_transfer_date": str(device.last_esim_transfer_date) if device.last_esim_transfer_date else None,
                "_device_capabilities": {
                    "supports_5g": device.model in ["Smartphone Pro Max", "iPhone 14", "Galaxy S23", "Pixel 7"],
                    "supports_wifi_calling": True,
                    "supports_volte": True,
                    "max_supported_bands": 42 if device.is_esim_capable else 32,
                },
                "_warranty_info": {
                    "manufacturer_warranty_status": "Active" if device.activated else "Pending",
                    "extended_warranty_eligible": device.is_esim_capable,
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        elif id.startswith("B"):
            bill = self._get_bill_by_id(id)
            
            # Build detailed line items
            line_items_detail = []
            subtotal_charges = 0.0
            subtotal_credits = 0.0
            for item in bill.line_items:
                item_dict = {
                    "description": item.description,
                    "amount": item.amount,
                    "date": str(item.date) if item.date else None,
                    "item_type": item.item_type,
                }
                if item.amount >= 0:
                    subtotal_charges += item.amount
                else:
                    subtotal_credits += abs(item.amount)
                line_items_detail.append(item_dict)
            
            return {
                "entity_type": "Bill",
                "bill_id": bill.bill_id,
                "customer_id": bill.customer_id,
                "period_start": str(bill.period_start),
                "period_end": str(bill.period_end),
                "issue_date": str(bill.issue_date),
                "due_date": str(bill.due_date),
                "total_due": bill.total_due,
                "status": bill.status.value,
                "line_items": line_items_detail,
                "_billing_breakdown": {
                    "subtotal_charges": subtotal_charges,
                    "subtotal_credits": subtotal_credits,
                    "taxes_and_fees": 0.0,  # Placeholder
                    "total_calculated": subtotal_charges - subtotal_credits,
                    "line_item_count": len(bill.line_items),
                },
                "_payment_info": {
                    "is_overdue": bill.status == BillStatus.OVERDUE,
                    "days_until_due": (bill.due_date - get_today()).days if bill.due_date else None,
                    "auto_pay_eligible": bill.status in [BillStatus.ISSUED, BillStatus.DRAFT],
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                    "billing_cycle": f"{bill.period_start.strftime('%B %Y')}" if bill.period_start else None,
                },
            }
        elif id.startswith("C"):
            # For customers, delegate to the enhanced get_customer_by_id
            customer = None
            for c in self.db.customers:
                if c.customer_id == id:
                    customer = c
                    break
            if not customer:
                raise ValueError(f"Customer with ID {id} not found")
                
            # Return same comprehensive format as get_customer_by_phone
            lines_summary = []
            total_data_used = 0.0
            active_lines = 0
            suspended_lines = 0
            
            for line_id in customer.line_ids:
                try:
                    line = self._get_line_by_id(line_id)
                    plan = self._get_plan_by_id(line.plan_id)
                    device = self._get_device_by_id(line.device_id)
                    total_data_used += line.data_used_gb
                    if line.status == LineStatus.ACTIVE:
                        active_lines += 1
                    elif line.status == LineStatus.SUSPENDED:
                        suspended_lines += 1
                        
                    lines_summary.append({
                        "line_id": line.line_id,
                        "phone_number": line.phone_number,
                        "status": line.status.value,
                        "plan_name": plan.name,
                        "device_model": device.model,
                        "data_used_gb": line.data_used_gb,
                        "roaming_enabled": line.roaming_enabled,
                    })
                except ValueError:
                    lines_summary.append({"line_id": line_id, "error": "Could not retrieve details"})
            
            bills_summary = []
            total_outstanding = 0.0
            for bill_id in customer.bill_ids:
                try:
                    bill = self._get_bill_by_id(bill_id)
                    bills_summary.append({
                        "bill_id": bill.bill_id,
                        "total_due": bill.total_due,
                        "status": bill.status.value,
                        "due_date": str(bill.due_date),
                    })
                    if bill.status in [BillStatus.OVERDUE, BillStatus.ISSUED]:
                        total_outstanding += bill.total_due
                except ValueError:
                    bills_summary.append({"bill_id": bill_id, "error": "Could not retrieve details"})
            
            return {
                "entity_type": "Customer",
                "customer_id": customer.customer_id,
                "full_name": customer.full_name,
                "date_of_birth": customer.date_of_birth,
                "email": customer.email,
                "phone_number": customer.phone_number,
                "account_status": customer.account_status.value,
                "address": {
                    "street": customer.address.street,
                    "city": customer.address.city,
                    "state": customer.address.state,
                    "zip_code": customer.address.zip_code,
                } if customer.address else None,
                "payment_methods": [
                    {
                        "method_type": pm.method_type,
                        "last_4": pm.account_number_last_4,
                        "expiration": pm.expiration_date,
                    }
                    for pm in (customer.payment_methods or [])
                ],
                "line_ids": customer.line_ids,
                "bill_ids": customer.bill_ids,
                "created_at": str(customer.created_at) if customer.created_at else None,
                "goodwill_credit_used_this_year": customer.goodwill_credit_used_this_year,
                "_lines_summary": lines_summary,
                "_billing_summary": {
                    "bills": bills_summary,
                    "total_outstanding": total_outstanding,
                },
                "_account_metrics": {
                    "total_lines": len(customer.line_ids),
                    "active_lines": active_lines,
                    "suspended_lines": suspended_lines,
                    "total_data_used_gb": round(total_data_used, 2),
                    "account_age_days": (get_today() - customer.created_at.date()).days if customer.created_at else None,
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        elif id.startswith("P"):
            plan = self._get_plan_by_id(id)
            
            # Calculate plan tier
            if plan.price_per_month <= 20:
                tier = "Entry"
            elif plan.price_per_month <= 50:
                tier = "Standard"
            elif plan.price_per_month <= 80:
                tier = "Premium"
            else:
                tier = "Enterprise"
            
            return {
                "entity_type": "Plan",
                "plan_id": plan.plan_id,
                "name": plan.name,
                "data_limit_gb": plan.data_limit_gb,
                "price_per_month": plan.price_per_month,
                "data_refueling_price_per_gb": plan.data_refueling_price_per_gb,
                "_plan_details": {
                    "tier": tier,
                    "annual_cost": plan.price_per_month * 12,
                    "cost_per_gb": round(plan.price_per_month / plan.data_limit_gb, 4) if plan.data_limit_gb > 0 else 0,
                    "includes_international_roaming": plan.data_limit_gb >= 15,
                    "includes_hotspot": plan.data_limit_gb >= 5,
                    "hd_streaming_included": plan.data_limit_gb >= 15,
                },
                "_comparison_metrics": {
                    "data_value_score": round(plan.data_limit_gb / plan.price_per_month * 10, 2) if plan.price_per_month > 0 else 0,
                    "refuel_affordability": "Low" if plan.data_refueling_price_per_gb >= 5 else ("Medium" if plan.data_refueling_price_per_gb >= 2 else "High"),
                },
                "_metadata": {
                    "query_id": id,
                    "query_timestamp": str(get_today()),
                    "system_version": "tau2-telecom-v2.1.0",
                },
            }
        else:
            raise ValueError(f"Unknown ID format or type: {id}")

    @is_tool(ToolType.WRITE)
    def suspend_line(
        self, customer_id: str, line_id: str, reason: str
    ) -> Dict[str, Any]:
        """
        Suspends a specific line (max 6 months).
        Checks: Line status must be Active.
        Logic: Sets line status to Suspended, records suspension_start_date.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to suspend.
            reason: Reason for suspension.

        Returns:
            Dictionary with success status, message, and updated line if applicable.

        Raises:
            ValueError: If customer or line not found, or if line is not active.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend")

        target_line.status = LineStatus.SUSPENDED
        target_line.suspension_start_date = get_today()

        # Log reason
        logger.info(f"Line {line_id} suspended. Reason: {reason}")

        return {
            "message": "Line suspended successfully. $5/month holding fee will apply.",
            "line": target_line,
        }

    @is_tool(ToolType.WRITE)
    def resume_line(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Resumes a suspended line.
        Checks: Line status must be Suspended or Pending Activation.
        Logic: Sets line status to Active, clears suspension_start_date.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to resume.

        Returns:
            Dictionary with success status, message, and updated line if applicable.

        Raises:
            ValueError: If customer or line not found, or if line is not suspended or pending activation.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.status not in [
            LineStatus.SUSPENDED,
            LineStatus.PENDING_ACTIVATION,
        ]:
            raise ValueError("Line must be suspended to resume")

        target_line.status = LineStatus.ACTIVE
        target_line.suspension_start_date = None

        # Log action
        logger.info(f"Line {line_id} resumed")

        return {
            "message": "Line resumed successfully",
            "line": target_line,
        }

    # Billing and Payments
    @is_tool(ToolType.READ)
    def get_bills_for_customer(self, customer_id: str, limit: int = 12) -> Dict[str, Any]:
        """
        Retrieves comprehensive billing history for a customer including
        all bills with details, payment history, and account balance analysis.

        Args:
            customer_id: ID of the customer.
            limit: Maximum number of bills to return.

        Returns:
            Dictionary with bills, summaries, and account status analysis.

        Raises:
            ValueError: If the customer is not found.
        """
        customer_data = self.get_customer_by_id(customer_id)
        # Extract customer from the enhanced response
        if isinstance(customer_data, dict):
            # Need to get the raw customer object for bill_ids
            customer = None
            for c in self.db.customers:
                if c.customer_id == customer_id:
                    customer = c
                    break
            if not customer:
                raise ValueError(f"Customer with ID {customer_id} not found")
        else:
            customer = customer_data

        bills = [self._get_bill_by_id(bill_id) for bill_id in customer.bill_ids]

        # Sort bills by issue date descending
        sorted_bills = sorted(bills, key=lambda bill: bill.issue_date, reverse=True)

        # Apply limit
        limited_bills = sorted_bills[:limit]
        
        # Build detailed bill list
        bills_detail = []
        total_paid = 0.0
        total_outstanding = 0.0
        total_overdue = 0.0
        overdue_bills = []
        
        for bill in limited_bills:
            line_items_summary = []
            for item in bill.line_items:
                line_items_summary.append({
                    "description": item.description,
                    "amount": item.amount,
                    "item_type": item.item_type,
                })
            
            bill_dict = {
                "bill_id": bill.bill_id,
                "period": f"{bill.period_start} to {bill.period_end}",
                "issue_date": str(bill.issue_date),
                "due_date": str(bill.due_date),
                "total_due": bill.total_due,
                "status": bill.status.value,
                "line_items": line_items_summary,
                "line_item_count": len(bill.line_items),
            }
            
            if bill.status == BillStatus.PAID:
                total_paid += bill.total_due
            elif bill.status == BillStatus.OVERDUE:
                total_overdue += bill.total_due
                total_outstanding += bill.total_due
                overdue_bills.append(bill.bill_id)
            elif bill.status in [BillStatus.ISSUED, BillStatus.AWAITING_PAYMENT]:
                total_outstanding += bill.total_due
            
            bills_detail.append(bill_dict)
        
        # Account health determination
        if total_overdue > 0:
            account_health = "AT_RISK"
            health_color = "RED"
        elif total_outstanding > 200:
            account_health = "ATTENTION_NEEDED"
            health_color = "YELLOW"
        else:
            account_health = "GOOD_STANDING"
            health_color = "GREEN"
        
        return {
            "customer_id": customer_id,
            "customer_name": customer.full_name,
            "bills": bills_detail,
            "bills_returned": len(bills_detail),
            "total_bills_on_account": len(customer.bill_ids),
            "_billing_summary": {
                "total_amount_paid": round(total_paid, 2),
                "total_amount_outstanding": round(total_outstanding, 2),
                "total_amount_overdue": round(total_overdue, 2),
                "overdue_bill_ids": overdue_bills,
                "has_overdue_bills": len(overdue_bills) > 0,
            },
            "_account_status": {
                "account_health": account_health,
                "health_indicator": health_color,
                "payment_methods_on_file": len(customer.payment_methods) if customer.payment_methods else 0,
                "auto_pay_enabled": False,  # Placeholder
            },
            "_payment_history": {
                "on_time_payment_rate": round(total_paid / (total_paid + total_overdue) * 100, 1) if (total_paid + total_overdue) > 0 else 100,
                "avg_bill_amount": round(sum(b.total_due for b in limited_bills) / len(limited_bills), 2) if limited_bills else 0,
            },
            "_recommendations": {
                "action_required": len(overdue_bills) > 0,
                "suggested_action": f"Pay overdue bills: {', '.join(overdue_bills)}" if overdue_bills else "No action required",
            },
            "_metadata": {
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
                "billing_cycle": "monthly",
            },
        }

    @is_tool(ToolType.WRITE)
    def send_payment_request(self, customer_id: str, bill_id: str) -> str:
        """
        Sends a payment request to the customer for a specific bill.
        Checks:
            - Customer exists
            - Bill exists and belongs to the customer
            - No other bills are already awaiting payment for this customer
        Logic: Sets bill status to AWAITING_PAYMENT and notifies customer.
        Warning: This method does not check if the bill is already PAID.
        Always check the bill status before calling this method.

        Args:
            customer_id: ID of the customer who owns the bill.
            bill_id: ID of the bill to send payment request for.

        Returns:
            Message indicating the payment request has been sent.

        Raises:
            ValueError: If customer not found, bill not found, or if another bill is already awaiting payment.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        bills = self._get_bills_awaiting_payment(customer)
        if len(bills) != 0:
            raise ValueError("A bill is already awaiting payment for this customer")
        if bill_id not in customer.bill_ids:
            raise ValueError(f"Bill {bill_id} not found for customer {customer_id}")
        bill = self._get_bill_by_id(bill_id)
        bill.status = BillStatus.AWAITING_PAYMENT
        return f"Payment request sent to the customer for bill {bill.bill_id}"

    def _get_bills_awaiting_payment(self, customer: Customer) -> List[Bill]:
        """
        Returns the bills in the customer's bill_ids list that are in the AWAITING_PAYMENT status.
        """
        bills = []
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.AWAITING_PAYMENT:
                bills.append(bill)
        return bills

    def _set_bill_to_paid(self, bill_id: str) -> None:
        """
        Sets the bill to paid.
        """
        bill = self._get_bill_by_id(bill_id)
        bill.status = BillStatus.PAID
        return f"Bill {bill_id} set to paid"

    def _apply_one_time_charge(
        self, customer_id: str, amount: float, description: str
    ) -> None:
        """
        Internal function to add a specific charge LineItem to the customer's next bill.
        Creates a pending bill if none exists.

        Args:
            customer_id: ID of the customer.
            amount: Amount to charge (positive) or credit (negative).
            description: Description of the charge.

        Returns:
            Success status.

        Raises:
            ValueError: If customer is not found (propagated from get_customer_by_id).
        """
        customer = self.get_customer_by_id(customer_id)
        # No need to check `if not customer`, get_customer_by_id raises if not found.

        # Find or create a draft bill
        draft_bill = None
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.DRAFT:
                draft_bill = bill
                break

        if not draft_bill:
            # Create a new draft bill for next cycle
            today = get_today()
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)  # First day of next month

            new_bill_id = f"B{uuid.uuid4().hex[:8]}"  # Simple ID generation
            draft_bill = Bill(
                bill_id=new_bill_id,
                customer_id=customer_id,
                period_start=next_month,
                period_end=next_month.replace(
                    month=next_month.month + 1 if next_month.month < 12 else 1,
                    year=(
                        next_month.year
                        if next_month.month < 12
                        else next_month.year + 1
                    ),
                )
                - timedelta(days=1),
                issue_date=next_month,
                total_due=0,
                due_date=next_month + timedelta(days=14),  # 14 days after issue
                status=BillStatus.DRAFT,
            )
            self.db.bills.append(draft_bill)
            customer.bill_ids.append(new_bill_id)

        # Add line item
        line_item = LineItem(
            description=description,
            amount=amount,
            date=get_today(),
            item_type="Credit" if amount < 0 else "Charge",
        )
        draft_bill.line_items.append(line_item)

        # Update total
        draft_bill.total_due += amount

    # Usage and Contract Info
    @is_tool(ToolType.READ)
    def get_data_usage(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Retrieves comprehensive data usage information for a line including
        current usage, limits, projections, refueling options, and historical context.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to check usage for.

        Returns:
            Dictionary with detailed usage information and analysis.

        Raises:
            ValueError: If customer, line, or plan not found.
        """
        target_line = self._get_target_line(customer_id, line_id)
        plan = self._get_plan_by_id(target_line.plan_id)
        device = self._get_device_by_id(target_line.device_id)

        today = get_today()
        cycle_start_date = date(today.year, today.month, 1)
        cycle_end_date = date(
            today.year, today.month + 1 if today.month < 12 else 1, 1
        ) - timedelta(days=1)
        
        days_in_cycle = (cycle_end_date - cycle_start_date).days + 1
        days_elapsed = (today - cycle_start_date).days + 1
        days_remaining = (cycle_end_date - today).days
        
        # Calculate usage metrics
        data_limit = plan.data_limit_gb
        data_used = target_line.data_used_gb
        data_refueled = target_line.data_refueling_gb
        effective_limit = data_limit + data_refueled
        data_remaining = max(0, effective_limit - data_used)
        utilization_pct = (data_used / effective_limit * 100) if effective_limit > 0 else 0
        
        # Projections
        daily_avg = data_used / days_elapsed if days_elapsed > 0 else 0
        projected_usage = daily_avg * days_in_cycle
        projected_overage = max(0, projected_usage - effective_limit)
        
        # Status determination
        if data_used >= effective_limit:
            usage_status = "EXCEEDED"
            status_color = "RED"
        elif utilization_pct >= 90:
            usage_status = "CRITICAL"
            status_color = "ORANGE"
        elif utilization_pct >= 75:
            usage_status = "WARNING"
            status_color = "YELLOW"
        else:
            usage_status = "NORMAL"
            status_color = "GREEN"

        return {
            "line_id": line_id,
            "phone_number": target_line.phone_number,
            "plan_name": plan.name,
            "device_model": device.model,
            "data_used_gb": data_used,
            "data_limit_gb": data_limit,
            "data_refueling_gb": data_refueled,
            "effective_data_limit_gb": effective_limit,
            "data_remaining_gb": round(data_remaining, 2),
            "_cycle_info": {
                "cycle_start_date": str(cycle_start_date),
                "cycle_end_date": str(cycle_end_date),
                "days_in_cycle": days_in_cycle,
                "days_elapsed": days_elapsed,
                "days_remaining": days_remaining,
                "cycle_progress_pct": round(days_elapsed / days_in_cycle * 100, 1),
            },
            "_usage_analysis": {
                "utilization_percentage": round(utilization_pct, 1),
                "daily_average_gb": round(daily_avg, 3),
                "projected_total_usage_gb": round(projected_usage, 2),
                "projected_overage_gb": round(projected_overage, 2),
                "usage_status": usage_status,
                "status_indicator": status_color,
            },
            "_overage_info": {
                "is_over_limit": data_used >= effective_limit,
                "overage_amount_gb": max(0, data_used - effective_limit),
                "overage_charge_rate": plan.data_refueling_price_per_gb,
                "current_overage_charge": max(0, (data_used - effective_limit) * plan.data_refueling_price_per_gb),
            },
            "_refueling_options": {
                "refueling_rate_per_gb": plan.data_refueling_price_per_gb,
                "suggested_refuel_1gb_cost": plan.data_refueling_price_per_gb * 1,
                "suggested_refuel_2gb_cost": plan.data_refueling_price_per_gb * 2,
                "suggested_refuel_5gb_cost": plan.data_refueling_price_per_gb * 5,
                "max_refuel_per_cycle_gb": 10.0,
            },
            "_recommendations": {
                "should_refuel": data_used >= effective_limit,
                "recommended_refuel_gb": min(5.0, max(0, data_used - data_limit + 2)) if data_used >= data_limit else 0,
                "consider_plan_upgrade": projected_overage > 5 and plan.price_per_month < 65,
            },
            "_metadata": {
                "customer_id": customer_id,
                "query_timestamp": str(get_today()),
                "system_version": "tau2-telecom-v2.1.0",
            },
        }

    def set_data_usage(
        self, customer_id: str, line_id: str, data_used_gb: float
    ) -> str:
        """
        Sets the data usage for a line.
        Note: This method is not decorated as a tool but follows similar error handling.

        Args:
            customer_id: ID of the customer.
            line_id: ID of the line.
            data_used_gb: Amount of data used in GB.

        Returns:
            Message indicating the data usage has been set.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        target_line.data_used_gb = data_used_gb
        return f"Data usage set to {data_used_gb} GB for line {line_id}"

    @is_tool(ToolType.WRITE)
    def enable_roaming(self, customer_id: str, line_id: str) -> Dict[str, Any]:
        """
        Enables international roaming on a line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to enable roaming for.

        Returns:
            Message indicating the roaming has been enabled.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if target_line.roaming_enabled:
            return "Roaming was already enabled"

        target_line.roaming_enabled = True

        logger.info(f"Roaming enabled for line {line_id}")

        return "Roaming enabled successfully"

    @is_tool(ToolType.WRITE)
    def disable_roaming(self, customer_id: str, line_id: str) -> str:
        """
        Disables international roaming on a line.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to disable roaming for.

        Returns:
            Message indicating the roaming has been enabled.

        Raises:
            ValueError: If customer or line not found.
        """
        target_line = self._get_target_line(customer_id, line_id)

        if not target_line.roaming_enabled:
            return "Roaming was already disabled"

        target_line.roaming_enabled = False

        logger.info(f"Roaming disabled for line {line_id}")

        return "Roaming disabled successfully"

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

    @is_tool(ToolType.WRITE)
    def refuel_data(
        self, customer_id: str, line_id: str, gb_amount: float
    ) -> Dict[str, Any]:
        """
        Refuels data for a specific line, adding to the customer's bill.
        Checks: Line status must be Active, Customer owns the line.
        Logic: Adds data to the line and charges customer based on the plan's refueling rate.

        Args:
            customer_id: ID of the customer who owns the line.
            line_id: ID of the line to refuel data for.
            gb_amount: Amount of data to add in gigabytes.

        Returns:
            Dictionary with success status, message, charge amount, and updated line if applicable.

        Raises:
            ValueError: If customer, line, or plan not found, or if checks fail.
        """
        target_line = self._get_target_line(customer_id, line_id)

        # if target_line.status != LineStatus.ACTIVE:
        #     raise ValueError("Line must be active to refuel data")

        if gb_amount <= 0:
            raise ValueError("Refuel amount must be positive")

        plan = self._get_plan_by_id(target_line.plan_id)
        if not plan:
            raise ValueError("Plan not found for this line")

        charge_amount = gb_amount * plan.data_refueling_price_per_gb

        target_line.data_refueling_gb += gb_amount

        self._apply_one_time_charge(
            customer_id,
            charge_amount,
            f"Data refueling: {gb_amount} GB at ${plan.data_refueling_price_per_gb}/GB",
        )

        logger.info(
            f"Data refueled for line {line_id}: {gb_amount} GB added, charge: ${charge_amount:.2f}"
        )

        return {
            "message": f"Successfully added {gb_amount} GB of data for line {line_id} for ${charge_amount:.2f}",
            "new_data_refueling_gb": target_line.data_refueling_gb,
            "charge": charge_amount,
        }

    ### Break tools
    def suspend_line_for_overdue_bill(
        self, customer_id: str, line_id: str, new_bill_id: str, contract_ended: bool
    ) -> str:
        """
        Suspends a line for an unpaid bill.
        """
        line = self._get_line_by_id(line_id)
        if line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend for unpaid bill")

        plan = self._get_plan_by_id(line.plan_id)
        amount = plan.price_per_month
        description = f"Charge for line {line.line_id}"

        if amount <= 0:
            raise ValueError("Amount must be positive for overdue bill")
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")

        overdue_bill_ids = []
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill.status == BillStatus.OVERDUE:
                overdue_bill_ids.append(bill_id)
        if len(overdue_bill_ids) > 0:
            raise ValueError("Customer already has an overdue bill")

        today = get_today()

        # Calculate the first day of the previous month using the same method as _apply_one_time_charge
        first_day_of_last_month = today.replace(day=1) - timedelta(days=1)
        first_day_of_last_month = first_day_of_last_month.replace(day=1)

        # Calculate the last day of the previous month
        last_day_of_last_month = today.replace(day=1) - timedelta(days=1)

        overdue_bill = Bill(
            bill_id=new_bill_id,
            customer_id=customer_id,
            period_start=first_day_of_last_month,
            period_end=last_day_of_last_month,
            issue_date=first_day_of_last_month,
            total_due=0,
            due_date=first_day_of_last_month + timedelta(days=14),
            status=BillStatus.OVERDUE,
        )
        line_item = LineItem(
            description=description,
            amount=amount,
            date=get_today(),
            item_type="Charge" if amount > 0 else "Credit",
        )
        overdue_bill.line_items.append(line_item)
        overdue_bill.total_due += amount
        self.db.bills.append(overdue_bill)
        customer.bill_ids.append(new_bill_id)
        line.status = LineStatus.SUSPENDED
        line.suspension_start_date = get_today()
        if contract_ended:
            line.contract_end_date = last_day_of_last_month
        return f"Line {line_id} suspended for unpaid bill {new_bill_id}. Contract ended: {contract_ended}"

    ### Assertions
    def assert_data_refueling_amount(
        self, customer_id: str, line_id: str, expected_amount: float
    ) -> bool:
        """
        Assert that the data refueling amount is as expected.
        """
        target_line = self._get_target_line(customer_id, line_id)
        return abs(target_line.data_refueling_gb - expected_amount) < 1e-6

    def assert_line_status(
        self, line_id: str, expected_status: str
    ) -> bool:
        """
        Assert that the line status is as expected.
        
        Args:
            line_id: The line ID to check.
            expected_status: Expected status as string (e.g., 'Active', 'Suspended').
        
        Returns:
            True if the line status matches expected.
        """
        target_line = self._get_line_by_id(line_id)
        return target_line.status.value == expected_status

    def assert_overdue_bill_exists(
        self, customer_id: str, overdue_bill_id: str
    ) -> bool:
        """
        Assert that the overdue bill exists.
        """
        customer = self.get_customer_by_id(customer_id)
        if not customer:
            raise ValueError(f"Customer {customer_id} not found")
        if overdue_bill_id not in customer.bill_ids:
            raise ValueError(f"Overdue bill {overdue_bill_id} not found")
        bill = self._get_bill_by_id(overdue_bill_id)
        if bill.status != BillStatus.OVERDUE:
            raise ValueError(f"Overdue bill {overdue_bill_id} is not overdue")
        return True

    def assert_no_overdue_bill(self, overdue_bill_id: str) -> bool:
        """
        Assert that either:
        - the overdue bill is not in the database
        - the overdue bill is paid
        """
        try:
            bill = self._get_bill_by_id(overdue_bill_id)
            if bill.status == BillStatus.PAID:
                return True
        except ValueError:
            return True
        return False

    def assert_bill_status(self, bill_id: str, expected_status: str) -> bool:
        """
        Assert that a bill has the expected status.
        
        Args:
            bill_id: The bill ID to check.
            expected_status: The expected status (e.g., 'Paid', 'Overdue', 'Issued').
        
        Returns:
            True if the bill status matches, False otherwise.
        """
        bill = self._get_bill_by_id(bill_id)
        return bill.status.value == expected_status

    def assert_line_plan(self, line_id: str, expected_plan_id: str) -> bool:
        """
        Assert that a line is on the expected plan.
        
        Args:
            line_id: The line ID to check.
            expected_plan_id: The expected plan ID (e.g., 'P1001', 'P1002').
        
        Returns:
            True if the line's plan matches, False otherwise.
        """
        line = self._get_line_by_id(line_id)
        return line.plan_id == expected_plan_id

    def assert_line_roaming_enabled(self, line_id: str, expected_enabled: bool) -> bool:
        """
        Assert that a line has roaming enabled/disabled as expected.
        
        Args:
            line_id: The line ID to check.
            expected_enabled: Whether roaming should be enabled (True) or disabled (False).
        
        Returns:
            True if the line's roaming status matches expected, False otherwise.
        """
        line = self._get_line_by_id(line_id)
        return line.roaming_enabled == expected_enabled


if __name__ == "__main__":
    from tau2.domains.telecom.utils import TELECOM_DB_PATH

    telecom = TelecomTools(TelecomDB.load(TELECOM_DB_PATH))
    print(telecom.get_statistics())
