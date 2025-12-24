"""
Simple Zoho API Integration - Working Version
"""

import os
import logging
import requests
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Import token manager for automatic refresh
from token_manager import get_token_manager


class ZohoSalesIQAPI:
    """Simple SalesIQ API Integration (Visitor API)"""
    
    def __init__(self):
        # Get token manager for automatic refresh
        self.token_manager = get_token_manager()
        
        # Load configuration
        self.access_token = self.token_manager.access_token
        self.department_id = os.getenv("SALESIQ_DEPARTMENT_ID", "").strip()
        self.app_id = os.getenv("SALESIQ_APP_ID", "").strip()
        self.screen_name = os.getenv("SALESIQ_SCREEN_NAME", "rtdsportal").strip()
        
        # Base URL for Visitor API v1 (official endpoint per API docs)
        self.base_url = f"https://salesiq.zoho.in/api/visitor/v1/{self.screen_name}"
        
        # Enable only if required config exists
        self.enabled = bool(self.access_token and self.department_id and self.app_id)
        if self.enabled:
            logger.info(f"SalesIQ Visitor API v1 configured - department: {self.department_id}, app_id: {self.app_id}, screen: {self.screen_name}")
        else:
            logger.warning(f"SalesIQ Visitor API not fully configured - token: {bool(self.access_token)}, dept: {bool(self.department_id)}, app_id: {bool(self.app_id)}, screen: {bool(self.screen_name)}")
    
    def create_chat_session(
        self,
        visitor_id: str,
        conversation_history: str,
        app_id: str | None = None,
        department_id: str | None = None,
        visitor_info: Dict | None = None,
        custom_wait_time: int | None = None,
    ) -> Dict:
        """Create a conversation via Visitor API to route to an agent.
        Allows overriding app_id/department_id/visitor_info from inbound webhook payload.
        
        IMPORTANT: Cannot use bot preview visitor IDs (botpreview_...).
        Must use real visitor IDs from actual chat widget interactions.
        """
        
        if not self.enabled:
            logger.info(f"SalesIQ: API disabled - simulating transfer for {visitor_id}")
            return {"success": True, "simulated": True, "message": "Transfer simulated"}
        
        # Reject bot preview IDs - they cannot be transferred
        if str(visitor_id).startswith("botpreview_"):
            logger.warning(f"SalesIQ: Cannot transfer bot preview session {visitor_id}. Need real visitor ID from actual chat widget.")
            return {
                "success": False,
                "error": "invalid_visitor_id",
                "details": "Bot preview sessions cannot be transferred. This is a SalesIQ limitation. Test with real visitor ID only."
            }
        
        import requests
        
        # Get fresh token (auto-refreshes if expired)
        valid_token = self.token_manager.get_valid_token()
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {valid_token}",
            "Content-Type": "application/json"
        }
        
        # Effective configuration (prefer overrides from webhook)
        effective_app_id = str(app_id or self.app_id).strip()
        effective_department_id = str(department_id or self.department_id).strip()

        # Extract visitor details from visitor_info or use defaults
        visitor_user_id = str(visitor_id).strip()
        visitor_name = "Chat User"
        visitor_email = "support@acecloudhosting.com"
        
        if visitor_info:
            # Prefer "email" as user_id if available (most reliable unique ID)
            visitor_user_id = visitor_info.get("email") or visitor_info.get("user_id") or visitor_user_id
            visitor_name = visitor_info.get("name") or visitor_info.get("email", "Chat User")
            visitor_email = visitor_info.get("email", "support@acecloudhosting.com")

        # Visitor API v1 payload structure per official documentation
        payload: Dict = {
            "app_id": effective_app_id,
            "department_id": effective_department_id,
            "question": conversation_history or "User requested human assistance",
            "visitor": {
                "user_id": visitor_user_id,
                "name": visitor_name,
                "email": visitor_email
            }
        }
        
        # Add optional custom_wait_time if provided
        if custom_wait_time is not None:
            payload["custom_wait_time"] = custom_wait_time
        
        endpoint = f"{self.base_url}/conversations"
        logger.info(f"SalesIQ: Visitor API v1 call - POST {endpoint}")
        logger.info(
            f"SalesIQ: Payload - app_id={effective_app_id}, dept={effective_department_id}, visitor_user_id={visitor_user_id}, visitor_email={visitor_email}"
        )
        
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            logger.info(f"SalesIQ: Response Status: {response.status_code}")
            logger.info(f"SalesIQ: Response Body: {response.text[:500]}")
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                except Exception:
                    data = {"raw": response.text}
                return {"success": True, "endpoint": endpoint, "data": data}
            else:
                return {"success": False, "error": f"{response.status_code}", "details": response.text}
        except Exception as e:
            logger.error(f"SalesIQ: Visitor API exception: {str(e)}")
            return {"success": False, "error": "exception", "details": str(e)}
    
    def close_chat(self, session_id: str, reason: str = "resolved") -> Dict:
        """Log chat closure"""
        
        logger.info(f"SalesIQ: Chat closure requested for {session_id}, reason: {reason}")
        
        return {
            "success": True,
            "message": f"Chat {session_id} closure logged"
        }


class ZohoDeskAPI:
    """Zoho Desk API Integration - Create tickets and callbacks"""
    
    def __init__(self):
        # Load configuration
        self.access_token = os.getenv("DESK_ACCESS_TOKEN", "").strip()
        self.organization_id = os.getenv("DESK_ORGANIZATION_ID", "").strip()
        self.api_url = os.getenv("DESK_API_URL", "https://desk.zoho.in/api/v1").strip()
        
        # Enable only if credentials exist
        self.enabled = bool(self.access_token and self.organization_id)
        
        if self.enabled:
            logger.info(f"Desk API configured - Org: {self.organization_id}")
        else:
            logger.warning(f"Desk API not configured - token: {bool(self.access_token)}, org_id: {bool(self.organization_id)}")
    
    def create_support_ticket(
        self,
        subject: str,
        description: str,
        user_email: str,
        phone: str = "",
        department: str = "Support",
        priority: str = "Medium",
        contact_name: str = ""
    ) -> dict:
        """Create a support ticket in Zoho Desk
        
        Args:
            subject: Ticket subject/title
            description: Detailed issue description
            user_email: Customer email
            phone: Customer phone number
            department: Department to assign ticket
            priority: Ticket priority (Low, Medium, High)
            contact_name: Customer name
        
        Returns:
            Dict with success status and ticket details
        """
        
        if not self.enabled:
            logger.info(f"Desk API disabled - simulating ticket creation for {user_email}")
            return {
                "success": True,
                "simulated": True,
                "ticket_number": "TK-SIM-001",
                "message": "Ticket creation simulated"
            }
        
        import requests
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.access_token}",
            "Content-Type": "application/json",
            "X-Orgn-Id": self.organization_id
        }
        
        # First, find or create the contact
        contact_id = self._find_or_create_contact(user_email, contact_name, phone)
        
        if not contact_id:
            logger.error(f"Desk: Failed to find/create contact for {user_email}")
            return {
                "success": False,
                "error": "contact_creation_failed",
                "details": f"Could not find or create contact for {user_email}"
            }
        
        # Create the ticket
        ticket_payload = {
            "subject": subject,
            "description": description,
            "contactId": contact_id,
            "priority": priority,
            "departmentId": department,
            "status": "Open"
        }
        
        endpoint = f"{self.api_url}/tickets"
        logger.info(f"Desk: Creating ticket - POST {endpoint}")
        logger.info(f"Desk: Subject: {subject}, Email: {user_email}")
        
        try:
            response = requests.post(
                endpoint,
                json=ticket_payload,
                headers=headers,
                timeout=10
            )
            
            logger.info(f"Desk: Response Status: {response.status_code}")
            logger.info(f"Desk: Response Body: {response.text[:500]}")
            
            if response.status_code in [200, 201]:
                try:
                    data = response.json()
                    ticket_id = data.get("id") or data.get("ticketNumber")
                    return {
                        "success": True,
                        "ticket_id": ticket_id,
                        "data": data
                    }
                except Exception:
                    return {
                        "success": True,
                        "message": "Ticket created successfully",
                        "raw_response": response.text
                    }
            else:
                return {
                    "success": False,
                    "error": f"{response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            logger.error(f"Desk: Ticket creation exception: {str(e)}")
            return {
                "success": False,
                "error": "exception",
                "details": str(e)
            }
    
    def create_callback_ticket(
        self,
        user_email: str,
        phone: str,
        preferred_time: str,
        contact_name: str = "",
        description: str = ""
    ) -> dict:
        """Create a callback ticket in Zoho Desk
        
        Args:
            user_email: Customer email
            phone: Customer phone number
            preferred_time: Preferred callback time
            contact_name: Customer name
            description: Callback reason/description
        
        Returns:
            Dict with success status and ticket details
        """
        
        if not self.enabled:
            logger.info(f"Desk API disabled - simulating callback ticket for {user_email}")
            return {
                "success": True,
                "simulated": True,
                "ticket_number": "CB-SIM-001",
                "message": "Callback ticket creation simulated"
            }
        
        # Create ticket with callback details
        subject = f"Callback Request - {contact_name or user_email}"
        callback_description = f"""
Callback Request
================
Email: {user_email}
Phone: {phone}
Preferred Time: {preferred_time}
Time Submitted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{description or 'Customer requested a callback from support team.'}
"""
        
        return self.create_support_ticket(
            subject=subject,
            description=callback_description,
            user_email=user_email,
            phone=phone,
            contact_name=contact_name,
            priority="Medium",
            department="Callbacks"
        )
    
    def _find_or_create_contact(
        self,
        email: str,
        name: str = "",
        phone: str = ""
    ) -> str:
        """Find existing contact or create new one
        
        Args:
            email: Contact email
            name: Contact name
            phone: Contact phone
        
        Returns:
            Contact ID if found/created, None otherwise
        """
        
        import requests
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {self.access_token}",
            "Content-Type": "application/json",
            "X-Orgn-Id": self.organization_id
        }
        
        # Search for existing contact by email
        search_endpoint = f"{self.api_url}/contacts"
        search_params = {
            "email": email
        }
        
        logger.info(f"Desk: Searching for contact: {email}")
        
        try:
            response = requests.get(
                search_endpoint,
                params=search_params,
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                contacts = data.get("data", [])
                
                if contacts and len(contacts) > 0:
                    contact_id = contacts[0].get("id")
                    logger.info(f"Desk: Found existing contact: {contact_id}")
                    return contact_id
        
        except Exception as e:
            logger.warning(f"Desk: Search failed: {str(e)}")
        
        # Contact not found - create new one
        logger.info(f"Desk: Contact not found - creating new contact")
        
        contact_payload = {
            "email": email,
            "firstName": name.split()[0] if name else "Customer",
            "lastName": name.split()[1] if name and len(name.split()) > 1 else "",
            "phone": phone if phone else ""
        }
        
        create_endpoint = f"{self.api_url}/contacts"
        
        try:
            response = requests.post(
                create_endpoint,
                json=contact_payload,
                headers=headers,
                timeout=10
            )
            
            logger.info(f"Desk: Contact creation status: {response.status_code}")
            
            if response.status_code in [200, 201]:
                data = response.json()
                contact_id = data.get("id")
                logger.info(f"Desk: New contact created: {contact_id}")
                return contact_id
            else:
                logger.error(f"Desk: Contact creation failed: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"Desk: Contact creation exception: {str(e)}")
            return None