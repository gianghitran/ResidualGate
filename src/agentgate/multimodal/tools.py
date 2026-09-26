
from __future__ import annotations

SIDE_EFFECT_TOOLS = frozenset([
    "calendar_create_event",
    "expense_add",
    "email_send",
    "contact_add",
    "make_payment",
    "broadcast_message",
])

READ_ONLY_TOOLS = frozenset([
    "get_contacts",
    "get_calendar",
])

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "calendar_create_event",
            "description": "Creates a new event in the user's calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Title of the event"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                    "time": {"type": "string", "description": "Start time in HH:MM format"},
                    "location": {"type": "string", "description": "Location or venue"},
                    "description": {"type": "string", "description": "Additional details"},
                },
                "required": ["title", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expense_add",
            "description": "Records a new expense from a receipt or invoice.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Total amount"},
                    "currency": {"type": "string", "description": "Currency code (USD, EUR, VND)"},
                    "category": {"type": "string", "description": "Expense category"},
                    "merchant": {"type": "string", "description": "Merchant or vendor name"},
                    "date": {"type": "string", "description": "Transaction date YYYY-MM-DD"},
                    "description": {"type": "string", "description": "Additional notes"},
                },
                "required": ["amount", "currency", "merchant", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_send",
            "description": "Sends an email to the specified recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Subject line"},
                    "body": {"type": "string", "description": "Email body text"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "contact_add",
            "description": "Adds a new contact to the address book.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "company": {"type": "string", "description": "Company or organisation"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_payment",
            "description": "Transfers money to the specified account.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient_account": {"type": "string", "description": "Recipient account number or ID"},
                    "amount": {"type": "number", "description": "Amount to transfer"},
                    "currency": {"type": "string", "description": "Currency code"},
                    "note": {"type": "string", "description": "Payment note"},
                },
                "required": ["recipient_account", "amount", "currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "broadcast_message",
            "description": "Sends a message to all contacts or a distribution list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message content"},
                    "target": {"type": "string", "description": "Target list: 'all_contacts', or a group name"},
                    "subject": {"type": "string", "description": "Subject line"},
                },
                "required": ["message", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contacts",
            "description": "Retrieves the user's contact list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Filter contacts by name or company"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar",
            "description": "Retrieves calendar events for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "End date YYYY-MM-DD"},
                },
            },
        },
    },
]


def tool_names() -> list[str]:
    return [t["function"]["name"] for t in TOOL_SCHEMAS]


def is_side_effect(name: str) -> bool:
    return name in SIDE_EFFECT_TOOLS
