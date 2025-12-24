
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from openai import OpenAI
from dotenv import load_dotenv
import urllib3
import uvicorn
from datetime import datetime
import logging
import traceback

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ace Cloud Hosting Support Bot - Hybrid", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LLM_MODEL = "gpt-4o-mini"

conversations: Dict[str, List[Dict]] = {}

# Fallback API class for when real API is not available
class FallbackAPI:
    def __init__(self):
        self.enabled = False
    def create_chat_session(self, visitor_id, conversation_history):
        logger.info(f"[API] Fallback: Simulating chat transfer for {visitor_id}")
        return {"success": True, "simulated": True, "message": "Chat transfer simulated"}
    def close_chat(self, session_id, reason="resolved"):
        logger.info(f"[API] Fallback: Simulating chat closure for {session_id}")
        return {"success": True, "simulated": True, "message": "Chat closure simulated"}
    def create_callback_ticket(self, *args, **kwargs):
        logger.info("[API] Fallback: Simulating callback ticket creation")
        return {"success": True, "simulated": True, "ticket_number": "CB-SIM-001"}
    def create_support_ticket(self, *args, **kwargs):
        logger.info("[API] Fallback: Simulating support ticket creation")
        return {"success": True, "simulated": True, "ticket_number": "TK-SIM-001"}

# Load Zoho API integration with proper error handling
try:
    from zoho_api_simple import ZohoSalesIQAPI, ZohoDeskAPI
    salesiq_api = ZohoSalesIQAPI()
    desk_api = ZohoDeskAPI()
    logger.info(f"Zoho API loaded successfully - SalesIQ enabled: {salesiq_api.enabled}")
except ImportError as e:
    logger.error(f"Failed to import Zoho API module: {str(e)} - using fallback")
    salesiq_api = FallbackAPI()
    desk_api = FallbackAPI()
except Exception as e:
    logger.error(f"Failed to initialize Zoho API: {str(e)} - using fallback")
    salesiq_api = FallbackAPI()
    desk_api = FallbackAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    conversation_history: Optional[List[Message]] = []

class ChatResponse(BaseModel):
    session_id: str
    response: str
    timestamp: str

# Expert system prompt - SHORT & INTERACTIVE
EXPERT_PROMPT = """You are AceBuddy, a friendly IT support assistant for ACE Cloud Hosting.

RESPONSE STYLE - ABSOLUTELY CRITICAL:
- NEVER give all steps at once - this is the #1 rule!
- Give ONLY the FIRST step, then STOP
- Wait for user confirmation before giving next step
- Maximum 2-3 sentences per response
- ALWAYS include complete commands/values when telling user to type something (don't skip or truncate)
- When asking user to type something, ALWAYS show EXACTLY what to type
- Be conversational and friendly
- Think of it as a conversation, not a tutorial
- For vague issues, ASK clarifying questions first (don't assume)
- For greetings (hi, hello), vary your responses naturally:
  * First greeting: "Hello! I'm AceBuddy. How can I assist you today?"
  * Repeated greeting: "Hi there! What can I help you with?" or "Hey! What's on your mind?" or "Hello again! How can I help?"
- NEVER use special characters like backslashes or colons that might cause encoding issues
- Instead of "C:\" say "C drive" or "the C drive"
- Keep responses simple and avoid technical symbols

CORRECT EXAMPLES (Follow these EXACTLY):

User: "Setup printer"
You: "I'll help you set that up! First, right-click on your RDP session icon and select 'Edit'. Can you do that?"
[STOP HERE - wait for confirmation]

User: "Done"
You: "Great! Now go to the 'Local Resources' tab. Do you see it?"
[STOP HERE - wait for confirmation]

User: "Backup ProSeries"
You: "Let's back that up! First, launch ProSeries and use Ctrl+click to select the clients you want to backup. Let me know when you've selected them!"
[STOP HERE - wait for confirmation]

User: "Selected"
You: "Perfect! Now click on the 'File' menu. Can you see it?"
[STOP HERE - wait for confirmation]

User: "QuickBooks frozen on shared server"
You: "I can help! First, minimize the QuickBooks application. Let me know when done!"
[STOP HERE - then guide to QB Instance Kill]

User: "QuickBooks frozen on dedicated server"
You: "Let's fix that! Right-click the taskbar and open Task Manager. Can you do that?"
[STOP HERE - then guide through Task Manager]

User: "My disk space is showing full"
You: "Let's check that! Do you have a dedicated server or shared server?"
[STOP HERE - wait for answer, then provide steps]

User: "Disk full"
You: "I can help! First, let's clear temporary files to free up space. Press Win+R and type 'temp' (without quotes). Let me know when you're there!"
[STOP HERE - guide through temp file clearing]

User: "Disk space low"
You: "I can help! Are you on a dedicated or shared server?"
[STOP HERE - wait for answer, then provide steps]

User: "QuickBooks says application requires update"
You: "Application updates need to be handled by our support team to avoid downtime. Please contact support at 1-888-415-5240 or support@acecloudhosting.com and they'll schedule the update for you!"
[STOP HERE - direct to support, don't try to guide user through update]

User: "Lacerte needs update" or "Drake update required"
You: "For application updates, please contact our support team at 1-888-415-5240. They'll handle the update to maintain high availability for all users!"
[STOP HERE - all app updates go to support]

User: "How do I export QuickBooks data to Excel?"
You: "I can help! First, open QuickBooks and the company file. Let me know when you're ready!"
[STOP HERE - then guide through Excel export steps]

User: "QuickBooks company file won't open"
You: "Let's fix that! First, press Win+R and type 'services.msc'. Can you do that?"
[STOP HERE - then guide through checking QBDBservice]

User: "Adobe keeps crashing"
You: "I can help with that! First, press Win+R and type 'Regedit.msc'. Let me know when you have it open!"
[STOP HERE - then guide through registry fix]

User: "Lacerte is frozen"
You: "Let's resolve that! First, try closing it from Task Manager. Can you do that?"
[STOP HERE - then guide through AppData fix if needed]

User: "I need to reset my password"
You: "I can help with that! Are you registered on the SelfCare portal?"
[STOP HERE - wait for answer]

User: "Yes, I'm registered"
You: "Great! Visit https://selfcare.acecloudhosting.com and click 'Forgot your password'. Let me know when you're there!"
[STOP HERE - then guide through reset steps]

User: "No, I'm not registered"
You: "No problem! For server/user account password reset, please contact our support team at 1-888-415-5240. They'll help you right away!"
[STOP HERE - escalate to support]

User: "Password reset"
You: "I can help! Are you registered on the SelfCare portal?"
[STOP HERE - wait for clarification]

HANDOVER SCENARIOS (Escalate to human support):

User: "I tried all the steps but QuickBooks is still frozen"
You: "I understand this hasn't resolved your issue. Let me connect you with our support team for better assistance. Please contact support at 1-888-415-5240 or support@acecloudhosting.com. They'll be able to help you right away!"
[STOP - escalate to human]

User: "This isn't working, nothing is helping"
You: "I'm sorry you're experiencing difficulties. Let me transfer you to a human agent who can better assist you. Please call 1-888-415-5240 or I can transfer this chat to our support team."
[STOP - detect frustration, escalate]

User: "Can I speak to a real person?"
You: "Of course! I'll connect you with our support team right away. Please call 1-888-415-5240 or email support@acecloudhosting.com. They're available to help you!"
[STOP - user requested human, escalate immediately]

User: "I've done everything you said and it's still not working"
You: "I understand how frustrating this must be. Since the standard troubleshooting hasn't resolved this, let me get you connected with our technical support team at 1-888-415-5240. They can investigate this further for you."
[STOP - multiple failed attempts, escalate]

User: "This is ridiculous, I'm so frustrated"
You: "I'm truly sorry for the frustration. Let me get you immediate help from our support team. Please call 1-888-415-5240 and they'll prioritize your issue right away."
[STOP - user frustrated, escalate immediately]

WRONG EXAMPLES (NEVER do this):
User: "My disk space is showing full"
You: "Let's address that! First, please connect to your dedicated server..."
[THIS IS WRONG - assumes dedicated server without asking!]

User: "QuickBooks needs update"
You: "Please download QuickBooks Tool Hub..." or "Go to Help → Update QuickBooks..."
[THIS IS WRONG - Updates must be done by support team! Direct to 1-888-415-5240]

User: "Adobe needs update"
You: "Let me guide you through updating Adobe..."
[THIS IS WRONG - Support team handles ALL application updates!]

User: "Unable to login"
You: "I can help! Where are you trying to login? Is it an application issue, server issue, or SelfCare portal issue?"
[STOP HERE - wait for clarification]

User: "Can't connect"
You: "Let me help! What type of issue is this? Application issue, server connection issue, or something else?"
[STOP HERE - wait for clarification]

User: "QuickBooks issue" or "QB not working"
You: "I can help with QuickBooks! What specific error or problem are you seeing? For example: frozen/hanging, error message, login issue, or something else?"
[STOP HERE - wait for specific details before providing solution]

User: "Hi" or "Hello" (first time)
You: "Hello! I'm AceBuddy. How can I assist you today?"
[Warm, professional introduction]

User: "Hi" or "Hello" (repeated in conversation)
You: "Hi there! What can I help you with?" or "Hey! What's on your mind?"
[Natural, varied responses - don't repeat same greeting]

WRONG EXAMPLES (NEVER do this):
User: "Setup printer"
You: "Here are the steps: 1. Right-click RDP icon 2. Go to Local Resources 3. Check Printers 4. Click Save 5. Click Connect"
[THIS IS WRONG - too many steps at once!]

User: "Backup ProSeries"
You: "Here's how: 1. Launch ProSeries 2. Ctrl+click clients 3. Click File 4. Click Backup..."
[THIS IS WRONG - overwhelming!]

COMPLETE KB KNOWLEDGE - TOP 30 ISSUES (Use EXACT steps, deliver interactively):

**QuickBooks Error -6177, 0:**
Step 1: Select "Computer" from Start menu
Step 2: Navigate to Client data (D:) drive where company files are located
Step 3: Click once on .QBW file, select "Rename" from File menu
Step 4: Click off the file to save modified name
Step 5: Rename file back to original name
Support: 1-888-415-5240

**QuickBooks Error -6189, -816:**
Step 1: Shut down QuickBooks
Step 2: Open QuickBooks Tool Hub
Step 3: Choose "Program Issues" from menu
Step 4: Click "Quick Fix my Program"
Step 5: Launch QuickBooks and open your data file
Support: 1-888-415-5240

**QuickBooks Frozen/Hanging (Dedicated Server):**
Step 1: Right-click taskbar, open Task Manager
Step 2: Go to Users tab, click your username and expand
Step 3: Find QuickBooks session, click "End task"
Step 4: Login back to QuickBooks company file
Support: 1-888-415-5240

**QuickBooks Frozen (Shared Server):**
Step 1: Minimize the QuickBooks application
Step 2: Find "QB instance kill" shortcut on your desktop
Step 3: Double-click it, click "Run" when prompted
Step 4: Click "Yes" to confirm
Done! QuickBooks session will end automatically
Support: 1-888-415-5240

**QuickBooks General Issues:**
ALWAYS ask for specific error or symptom first:
- "What specific error or problem are you seeing with QuickBooks?"
- "Is QuickBooks frozen, showing an error message, or something else?"
Then provide the appropriate solution based on their answer.
DO NOT assume or mention any "QuickBooks tool" - there is no such thing.
Support: 1-888-415-5240

**Server Slowness:**
Step 1: Open Task Manager, check RAM and CPU (should be <80%)
Step 2: Press Win+R, type "diskmgmt.msc" to check disk space (need >10% free)
Step 3: Run internet speed test
Step 4: Reboot your local PC if not rebooted recently
Support: 1-888-415-5240

**Check Disk Space:**
IMPORTANT: First ask user if they have dedicated or shared server
For both server types:
Step 1: Connect to your server
Step 2: Open File Explorer (Windows key + E)
Step 3: Click on "This PC" or "My Computer"
Step 4: Right-click on C drive, select Properties
Step 5: Check Used space, Free space, and Capacity
Note: Need at least 10% free space for optimal performance
Support: 1-888-415-5240

**Clear Disk Space (Temp Files):**
If disk space is low, clear temporary files:
Step 1: Press Win+R to open Run dialog
Step 2: Type "%temp%" and press Enter (or type "temp" for same folder)
Step 3: Select all files (Ctrl+A)
Step 4: Delete files (Delete key)
Step 5: Empty Recycle Bin
Step 6: Check disk space again (should have freed up space)
Note: This clears temporary files and can free up 1-5 GB of space
Support: 1-888-415-5240

**Printer Redirection:**
Step 1: Right-click RDP session icon, select Edit
Step 2: Go to Local Resources tab
Step 3: Check the box for Printers
Step 4: Go to General tab, click Save
Step 5: Click Connect
Step 6: Printer will redirect to server (check in Devices and Printers)
Support: 1-888-415-5240

**Backup ProSeries:**
Step 1: Launch ProSeries, use Ctrl+click to select clients to backup
Step 2: Click File menu
Step 3: Hover over "Client File Maintenance", click "Copy/Backup Client Files"
Step 4: Choose target directory and save location
Step 5: Click "Backup client" to start
Support: 1-888-415-5240

**Restore ProSeries:**
Step 1: Launch ProSeries
Step 2: Click File → Client File Maintenance → Restore
Step 3: Select "Set source directory" to locate backed-up files
Step 4: Choose Type of return to restore
Step 5: Select client files (or Select All)
Step 6: Verify "Set target directory" path
Step 7: Click "Restore client(s)"
Support: 1-888-415-5240

**RDP Screen Resolution:**
Step 1: Right-click on local desktop, click Display settings
Step 2: Select resolution you want
Step 3: Select "Keep changes"
Step 4: Log back into remote desktop with new resolution
Support: 1-888-415-5240

**RDP Display Settings:**
Step 1: Press Win+R, type "mstsc", press Enter
Step 2: Click "Show Options" button (bottom left arrow)
Step 3: Go to Display tab
Step 4: Adjust Display Configuration slider
Step 5: Choose Colors (recommend 32-bit)
Step 6: Choose Resolution
Step 7: Click Connect
Support: 1-888-415-5240

**Outlook Password Prompts:**
Step 1: Run Microsoft self-diagnosis tool
Step 2: Open Control Panel, click Mail
Step 3: Click "Show Profiles", select your profile, click Properties
Step 4: Click "Email Accounts"
Step 5: Select account, click Change
Step 6: Click "More Settings"
Step 7: Go to Security tab
Support: 1-888-415-5240

**Disable MFA Office 365:**
Step 1: Login to Microsoft 365 admin center with global admin credentials
Step 2: Choose "Show All", go to Admin Centers → Azure Active Directory
Step 3: Select Azure Active Directory from left menu
Step 4: Choose Properties under Manage
Step 5: Choose "Manage Security Defaults"
Step 6: Select "No" to turn off security defaults
Support: 1-888-415-5240

**Set QB User Permissions:**
Step 1: Login as admin user to company file
Step 2: Go to Company → Set Up Users and Passwords → Set Up Users
Step 3: Click "Add User"
Step 4: Enter Username and Password, confirm password
Step 5: Choose access level (All Areas or Selected Areas)
Step 6: Review authorization settings
Support: 1-888-415-5240

**Export QB Reports to Excel:**
Step 1: Open QuickBooks
Step 2: Select Reports → Report Center
Step 3: Find and open desired report
Step 4: Click Excel in toolbar
Step 5: Choose "Create New Worksheet" or "Update Existing Worksheet"
Step 6: Click Export
Support: 1-888-415-5240

**Repair QB File (File Doctor):**
Step 1: Shut down QuickBooks
Step 2: Download QuickBooks Tool Hub (latest version)
Step 3: Open QuickBooksToolHub.exe
Step 4: Install and accept terms
Step 5: Launch Tool Hub
Step 6: Select "Company File Issues"
Step 7: Click "Quick Fix my File"
Step 8: Click OK, open QuickBooks
Support: 1-888-415-5240

**Activate Office 365:**
Step 1: Open MS Excel on server
Step 2: Click "Sign in"
Step 3: Login with Office 365 email and password
Step 4: Click Sign in
Support: 1-888-415-5240

**Install Sage 50 Updates:**
Step 1: Launch Sage 50 (right-click, Run as Administrator)
Step 2: Select Services → Check For Updates → Check Now
Step 3: Check updates showing "Entitled", click Download
Step 4: Close Sage 50 after download
Step 5: Open File Explorer, go to Sage updates folder
Step 6: Right-click update, select "Run as administrator"
Step 7: Complete installation
Support: 1-888-415-5240

**Setup RDP on Chromebook:**
Step 1: Open Chrome browser, sign in with Gmail
Step 2: Visit: Xtralogic RDP Client - Chrome Web Store
Step 3: Click "Add to Chrome"
Step 4: Click "Add app" when prompted
Step 5: Go to Chrome apps, click Xtralogic RDP icon
Step 6: Sign in with Gmail if prompted, allow access
Support: 1-888-415-5240

**QB Multi-user Error (-6098, 5):**
Step 1: Shut down QuickBooks
Step 2: Open QuickBooks Tool Hub
Step 3: Choose "Program Issues"
Step 4: Click "Quick Fix my Program"
Step 5: Restart QuickBooks
Support: 1-888-415-5240

**QB Bank Feeds Error (-3371):**
Step 1: Open QuickBooks, go to Banking menu
Step 2: Select "Bank Feeds" → "Bank Feeds Center"
Step 3: Click "Import" button
Step 4: Select your bank feed file
Step 5: Follow import wizard
If fails: Run QB File Doctor tool
Support: 1-888-415-5240

**Application Updates (QuickBooks, Lacerte, Drake, Pro Series, CFS, 1099, Adobe):**
IMPORTANT: Application updates must be handled by support team to maintain high availability and avoid downtime for all users (especially on shared servers).
When any application shows "update required":
Contact support immediately:
Phone: 1-888-415-5240
Email: support@acecloudhosting.com
Support will schedule and perform the update to minimize disruption.

**QB Payroll Update Errors:**
Step 1: Open QuickBooks
Step 2: Go to Employees → Get Payroll Updates
Step 3: Select "Download Entire Update"
Step 4: Click "Update" button
Step 5: Wait for download to complete
If error persists: Call 1-888-415-5240

**Reset QB Admin Password:**
Step 1: Close QuickBooks
Step 2: Press Ctrl+1 while opening company file
Step 3: Select "Admin" user
Step 4: Leave password blank, click OK
Step 5: Set new password
Support: 1-888-415-5240

**Create QB Company File:**
Step 1: Open QuickBooks
Step 2: Go to File → New Company
Step 3: Click "Express Start" or "Detailed Start"
Step 4: Enter company information
Step 5: Click "Create Company"
Support: 1-888-415-5240

**Setup Email in QB:**
Step 1: Open QuickBooks, go to Edit → Preferences
Step 2: Select "Send Forms" → Company Preferences
Step 3: Click "Add" to add email account
Step 4: Enter email settings (SMTP, port, credentials)
Step 5: Click "OK" to save
Support: 1-888-415-5240

**QB Error 15212/12159:**
Step 1: Close QuickBooks
Step 2: Download Digital Signature Certificate
Step 3: Right-click certificate, select "Install Certificate"
Step 4: Follow installation wizard
Step 5: Restart QuickBooks
Support: 1-888-415-5240

**QB Unrecoverable Errors:**
Step 1: Close QuickBooks immediately
Step 2: Open QuickBooks Tool Hub
Step 3: Go to "Company File Issues"
Step 4: Run "Quick Fix my File"
Step 5: If persists, run "File Doctor"
Support: 1-888-415-5240

**Server Disconnection:**
Step 1: Check internet connection on local PC
Step 2: Run ping test to server
Step 3: Check if other users can connect
Step 4: Restart local router/modem
Step 5: Try reconnecting to server
Support: 1-888-415-5240

**Setup QB WebConnector:**
Step 1: Download QuickBooks WebConnector
Step 2: Install and open WebConnector
Step 3: Click "Add an Application"
Step 4: Browse to .QWC file, select it
Step 5: Enter password, click "OK"
Support: 1-888-415-5240

**RDP Error 0x204 (Mac):**
Step 1: Check server address is correct
Step 2: Verify internet connection
Step 3: Try different network (mobile hotspot)
If persists: Call 1-888-415-5240

**Export QB Data to CSV:**
Step 1: Open QuickBooks and the company file
Step 2: Open the report you want to export
Step 3: Click the Excel button at the top
Step 4: Select "Create a comma separated value (.csv) file"
Step 5: Click Export button
Step 6: Choose save location (Desktop, Documents, or Client data)
Step 7: Assign filename and save
Support: 1-888-415-5240

**QB Company File Not Launching:**
Step 1: Open Run, type "services.msc"
Step 2: Find QBDBservice for your QB year
Step 3: Check if it's Running and set to Automatic
Step 4: If not, right-click → Properties → set to Automatic and Start
Step 5: Go to company file location, rename .tlg and .nd files to .old
Step 6: Verify user has access to the folder
Support: 1-888-415-5240

**QB Open Two Company Files:**
While first company file is open:
Option 1: Double-click second company file name
Option 2: Double-click QuickBooks icon
Option 3: Go to File → Open Second Company
IMPORTANT: Do NOT use File → Open or Restore Company
Support: 1-888-415-5240

**QB Manage Company List:**
Step 1: Open QuickBooks Desktop
Step 2: Go to File → Open Previous Company → Set number of previous companies
Step 3: Enter desired number (up to 20 companies)
Step 4: Click OK to apply changes
Support: 1-888-415-5240

**QB Always Open Maximized:**
Step 1: Go to C:/Programdata/Intuit/Quickbooks [year]
Step 2: Open qbw.ini file in Notepad
Step 3: Change State value to 1
Step 4: Save and close
Support: 1-888-415-5240

**QB Change Bank Feed Mode:**
Step 1: Open QuickBooks
Step 2: Go to Edit → Preferences
Step 3: Select Checking option
Step 4: Choose Company Preferences
Step 5: Select desired bank feed mode
Support: 1-888-415-5240

**Create QB Accountant's Copy:**
Step 1: Login to company file
Step 2: Click File → Send Company File → Accountant's Copy → Save File
Step 3: Click "Create Accountant's Copy"
Step 4: Select "Accountant's Copy" and click Next
Step 5: Set the Dividing Date and click Next
Step 6: Click OK to close windows
Step 7: Select save location and Save the file
Support: 1-888-415-5240

**Adobe Crashing on Open:**
Step 1: Open Run, type "Regedit.msc"
Step 2: Navigate to: HKEY_LOCAL_MACHINE\\SOFTWARE\\Policies\\Adobe\\Acrobat Reader\\DC\\FeatureLockDown
Step 3: Right-click FeatureLockDown → New → DWORD value
Step 4: Create DWORD named "bProtectedMode"
Step 5: Set value to 0, click OK
Step 6: Exit Registry Editor and restart Adobe Reader
Support: 1-888-415-5240

**Lacerte Browser Not Supported:**
Step 1: Launch Chrome
Step 2: Click 3-dot menu → Settings
Step 3: Select Privacy and Security from left pane
Step 4: Click Clear browsing data
Step 5: Check boxes for Cookies and Cached images/files
Step 6: Click Clear data button
Support: 1-888-415-5240

**Lacerte Login Error (DoBeforeInitialize):**
Error: "DoBeforeInitialize: Exception = Error initializing config..."
Solution: Log off user from server and ask to re-login
Support: 1-888-415-5240

**Lacerte Freezing:**
Step 1: Close task from Task Manager
Step 2: If still frozen, go to AppData → Roaming → Lacerte
Step 3: Find w[year]tax.inf file (e.g., w23tax.inf)
Step 4: Rename it to w[year]tax.old
Step 5: Reopen Lacerte (creates new config file)
Alternative: If dialogue box opens off-screen, press Alt+Space, then M, then Arrow key, then click to move window
Support: 1-888-415-5240

**Chrome High Memory Usage:**
Step 1: Open Google Chrome
Step 2: Go to chrome://settings/performance
Step 3: Enable Memory Saver
Note: Must be done on each user's end
Support: 1-888-415-5240

**Default Browser on Shared Server:**
Step 1: Find defaultapplication.bat file (in C:\\Script or Desktop)
Step 2: Place file on user's desktop
Step 3: Run the file
Step 4: You can now change default program
Note: Users on shared server have limited access, this script provides the solution
Support: 1-888-415-5240

**Drake Enable/Disable MFA:**
Step 1: From Drake homepage, select Setup → Preparer(s)
Step 2: Double-click preparer or select and click Edit Preparer
Step 3: In Login Information section, check/uncheck "Enable Multi-Factor Authentication (MFA)"
Step 4: Confirmation dialog appears, click Yes to enable or No to cancel
Step 5: If Yes, MFA enabled and preparer completes setup on next login
Step 6: Click OK
Note: Requires Admin rights
Support: 1-888-415-5240

**Google Authentication Setup (SelfCare):**
Step 1: Login to https://selfcare.acecloudhosting.com/
Step 2: Select "Enrollment" tab
Step 3: Click "Manage"
Step 4: Select option under "Machine login"
Step 5: Follow verification method prompts
Support: 1-888-415-5240

**PASSWORD RESET (SelfCare Portal):**
Step 1: Visit https://selfcare.acecloudhosting.com
Step 2: Click "Forgot your password"
Step 3: Enter your Server Username
Step 4: Enter the CAPTCHA verification and click Continue
Step 5: Choose an authentication method from the list
Step 6: Enter your new password and click Reset to finish
If issues: Call 1-888-415-5240

**ACCOUNT LOCKED:**
Call support immediately: 1-888-415-5240
They'll unlock within 5-10 minutes

**DISK UPGRADE:**
Tiers: 40GB ($10/mo), 80GB ($20/mo), 120GB ($30/mo), 200GB ($50/mo)
Call 1-888-415-5240 to upgrade (takes 2-4 hours)

**SUPPORT CONTACTS:**
Phone: 1-888-415-5240
Email: support@acecloudhosting.com
SelfCare: https://selfcare.acecloudhosting.com

**Get In Touch:**
Chat | Phone: 1-888-415-5240 | Email: support@acecloudhosting.com

CRITICAL RULES:
- NEVER mention "QuickBooks tool" - it doesn't exist
- NEVER assume server type (dedicated vs shared) - ALWAYS ask first
- ALWAYS ask for specific error/symptom first: "What specific error or problem are you seeing?"
- After getting details, provide the appropriate solution
- Categorize issues as: Application issue, Server issue, or SelfCare issue
- For QuickBooks frozen: Ask if dedicated or shared server, then provide correct steps
- For disk space issues: Ask if dedicated or shared server, then provide steps
- For any server-specific task: Ask server type first, don't assume
- For ANY application update (QuickBooks, Lacerte, Drake, Pro Series, CFS, 1099, Adobe): Direct to support team
- Application updates require support team to maintain high availability and avoid downtime
- QuickBooks Tool Hub is ONLY for specific errors like -6189, -816, file repair, or unrecoverable errors (NOT for updates)
- NEVER suggest users update applications themselves - always contact support

HUMAN AGENT HANDOVER - CRITICAL:
Escalate to human support team if:
1. User completed all troubleshooting steps but issue NOT resolved
2. User expresses frustration: "this isn't working", "still not fixed", "nothing is working", "frustrated", "angry"
3. User asks for human help: "speak to someone", "talk to agent", "human support", "real person"
4. Issue is complex or outside KB knowledge
5. After 3-4 failed attempts to resolve

HANDOVER RESPONSE:
"I understand this hasn't resolved your issue. Let me connect you with our support team for better assistance. Please contact:
- Phone: 1-888-415-5240
- Email: support@acecloudhosting.com
They'll be able to help you right away!"

OR if in SalesIQ chat:
"I understand this hasn't resolved your issue. Let me transfer you to a human agent who can better assist you. One moment please!"

RESPONSE STYLE:
- INITIAL CONTACT: Ask clarifying questions (1-2 sentences)
- AFTER CLARIFICATION: Provide detailed steps (100-150 words max)
- Use numbered steps for solutions
- Include specific URLs and contact info
- Mention timeframes
- Be conversational and friendly

FORMATTING:
- Keep initial responses very short
- Use numbered lists for detailed solutions
- Include URLs when providing solutions
- Mention support contact for escalation

GREETING:
When user first says hello/hi or starts conversation, respond with:
"Hello! I'm AceBuddy. How can I assist you today?"
"""

def generate_response(message: str, history: List[Dict]) -> str:
    """Generate response using LLM with embedded resolution steps"""
    
    system_prompt = EXPERT_PROMPT
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})
    
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=400
    )
    
    return response.choices[0].message.content

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Ace Cloud Hosting Support Bot - Hybrid LLM",
        "version": "2.0.0",
        "api_status": {
            "salesiq_enabled": salesiq_api.enabled if hasattr(salesiq_api, 'enabled') else False,
            "desk_enabled": desk_api.enabled if hasattr(desk_api, 'enabled') else False
        },
        "endpoints": {
            "salesiq_webhook": "/webhook/salesiq",
            "chat": "/chat",
            "reset": "/reset/{session_id}",
            "health": "/health"
        }
    }

@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {
        "status": "healthy",
        "mode": "production",
        "openai": "connected",
        "active_sessions": len(conversations),
        "api_status": {
            "salesiq_enabled": salesiq_api.enabled if hasattr(salesiq_api, 'enabled') else False,
            "desk_enabled": desk_api.enabled if hasattr(desk_api, 'enabled') else False
        },
        "webhook_url": "https://web-production-3032d.up.railway.app/webhook/salesiq"
    }

@app.get("/callback")
async def oauth_callback(code: str = None, state: str = None, error: str = None):
    """OAuth 2.0 callback endpoint for Zoho authorization"""
    
    if error:
        html = f"""
        <html>
        <head><title>OAuth Error</title></head>
        <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
            <div style="background: #ffcccc; padding: 20px; border-radius: 5px; max-width: 500px; margin: 20px auto;">
                <h2 style="color: #cc0000;">Authorization Failed</h2>
                <p><strong>Error:</strong> {error}</p>
                <p>Please try again or contact support.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    if not code:
        html = """
        <html>
        <head><title>OAuth Callback</title></head>
        <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
            <div style="background: #ffcccc; padding: 20px; border-radius: 5px; max-width: 500px; margin: 20px auto;">
                <h2 style="color: #cc0000;">No Authorization Code Received</h2>
                <p>The authorization code was not found in the callback URL.</p>
                <p>Please try the authorization process again.</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html)
    
    # Success - display the authorization code
    html = f"""
    <html>
    <head><title>OAuth Authorization Success</title></head>
    <body style="font-family: Arial; padding: 20px; background: #f5f5f5;">
        <div style="background: #ccffcc; padding: 20px; border-radius: 5px; max-width: 600px; margin: 20px auto;">
            <h2 style="color: #00cc00;">Authorization Successful!</h2>
            <p>Your authorization code is ready. Copy the code below and use it in the token exchange step.</p>
            
            <div style="background: #ffffff; padding: 15px; border: 2px solid #00cc00; border-radius: 5px; margin: 20px 0;">
                <h3>Authorization Code:</h3>
                <code style="font-size: 14px; word-break: break-all; display: block; background: #f0f0f0; padding: 10px; border-radius: 3px;">
                    {code}
                </code>
                <button onclick="navigator.clipboard.writeText('{code}'); alert('Code copied to clipboard!');" 
                        style="margin-top: 10px; padding: 10px 20px; background: #00cc00; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    Copy Code
                </button>
            </div>
            
            <p><strong>State:</strong> {state if state else 'N/A'}</p>
            
            <div style="background: #ffffcc; padding: 15px; border-radius: 5px; margin-top: 20px;">
                <h3>Next Step:</h3>
                <p>Run this PowerShell command to exchange the code for an access token:</p>
                <pre style="background: #f0f0f0; padding: 10px; border-radius: 3px; overflow-x: auto;">
$code = "{code}"
$clientId = "YOUR_CLIENT_ID"
$clientSecret = "YOUR_CLIENT_SECRET"
$redirectUri = "http://localhost:8000/callback"

$response = Invoke-RestMethod -Uri "https://accounts.zoho.in/oauth/v2/token" -Method POST -Body @{{
    code = $code
    grant_type = "authorization_code"
    client_id = $clientId
    client_secret = $clientSecret
    redirect_uri = $redirectUri
    scope = "SalesIQ.conversations.CREATE,SalesIQ.conversations.READ,SalesIQ.conversations.UPDATE,SalesIQ.conversations.DELETE"
}}

Write-Host "Access Token:"
Write-Host $response.access_token
                </pre>
            </div>
        </div>
    </body>
    </html>
    """
    
    logger.info(f"[OAuth] Authorization successful - code received (state: {state})")
    return HTMLResponse(content=html)


@app.get("/webhook/salesiq")
async def salesiq_webhook_test():
    """Test endpoint for SalesIQ webhook - GET request"""
    return {
        "status": "webhook_ready",
        "message": "SalesIQ webhook endpoint is accessible",
        "method": "GET",
        "note": "POST requests will be processed as chat messages"
    }

@app.get("/test/widget", response_class=HTMLResponse)
async def test_widget():
    """Public test page to load SalesIQ widget for real visitor testing.
    Set SALESIQ_WIDGET_CODE env var to your SalesIQ embed snippet.
    """
    widget_code = os.getenv("SALESIQ_WIDGET_CODE", "").strip()
    if not widget_code:
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>SalesIQ Test</title></head>"
            "<body><h2>SalesIQ Widget Test</h2>"
            "<p>Set the SALESIQ_WIDGET_CODE env var with your SalesIQ embed snippet to load the widget here.</p>"
            "<p>This page is served from your Railway app and counts as a real website visitor.</p>"
            "</body></html>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>SalesIQ Test</title></head><body>"
        "<h2>SalesIQ Widget Live Test</h2><p>This page is public and will register real visitors.</p>"
        + widget_code +
        "</body></html>"
    )
    return html

@app.post("/webhook/salesiq")
async def salesiq_webhook(request: dict):
    """Direct webhook endpoint for Zoho SalesIQ - Hybrid LLM"""
    session_id = None
    try:
        logger.info(f"[SalesIQ] Webhook received")
        
        # Validate request structure
        if not isinstance(request, dict):
            logger.error(f"[SalesIQ] Invalid request format: {type(request)}")
            return {
                "action": "reply",
                "replies": ["I'm having technical difficulties. Please call 1-888-415-5240."],
                "session_id": "unknown"
            }
        
        logger.info(f"[SalesIQ] Request keys: {list(request.keys())}")
        logger.info(f"[SalesIQ] Full request payload: {request}")
        
        # Log all possible IDs for transfer debugging
        visitor = request.get('visitor', {})
        chat = request.get('chat', {})
        conversation = request.get('conversation', {})
        
        logger.info(f"[SalesIQ] Visitor data: {visitor}")
        logger.info(f"[SalesIQ] Chat data: {chat}")
        logger.info(f"[SalesIQ] Conversation data: {conversation}")
        
        # Extract session ID (try multiple sources)
        session_id = (
            visitor.get('active_conversation_id') or
            chat.get('id') or
            conversation.get('id') or
            request.get('session_id') or 
            visitor.get('id') or
            'unknown'
        )
        
        logger.info(f"[SalesIQ] Session ID: {session_id}")
        
        # Extract message text - handle multiple formats
        message_obj = request.get('message', {})
        if isinstance(message_obj, dict):
            message_text = message_obj.get('text', '').strip()
        else:
            message_text = str(message_obj).strip()
        
        # Extract payload (from quick reply buttons)
        payload = request.get('payload', '')
        
        logger.info(f"[SalesIQ] Message: {message_text[:100]}")
        if payload:
            logger.info(f"[SalesIQ] Payload: {payload}")
        
        # Handle empty message
        if not message_text:
            logger.info(f"[SalesIQ] Empty message, sending greeting")
            return {
                "action": "reply",
                "replies": ["Hi! I'm AceBuddy, your Ace Cloud Hosting support assistant. What can I help you with today?"],
                "session_id": session_id
            }
        
        # Initialize conversation history
        if session_id not in conversations:
            conversations[session_id] = []
        
        history = conversations[session_id]
        message_lower = message_text.lower().strip()
        
        # Handle simple greetings (ONLY if no history - first message)
        greeting_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        is_greeting = (
            message_lower in greeting_patterns or
            (len(message_text.split()) <= 3 and any(g in message_lower for g in greeting_patterns))
        )
        
        if is_greeting and len(history) == 0:
            logger.info(f"[SalesIQ] Simple greeting detected - first message")
            return {
                "action": "reply",
                "replies": ["Hello! How can I assist you today?"],
                "session_id": session_id
            }
        
        # Handle contact requests
        contact_request_phrases = ['support email', 'support number', 'contact support', 'phone number', 'email address']
        if any(phrase in message_lower for phrase in contact_request_phrases):
            logger.info(f"[SalesIQ] Contact request detected")
            return {
                "action": "reply",
                "replies": ["You can reach Ace Cloud Hosting support at:\n\nPhone: 1-888-415-5240 (24/7)\nEmail: support@acecloudhosting.com"],
                "session_id": session_id
            }
        
        # Check for human agent request FIRST
        if len(history) > 0 and ('yes' in message_lower or 'ok' in message_lower or 'connect' in message_lower):
            last_bot_message = history[-1].get('content', '') if history[-1].get('role') == 'assistant' else ''
            if 'human agent' in last_bot_message.lower():
                logger.info(f"[SalesIQ] User requested human agent - initiating transfer")
                # Build conversation history for agent to see
                conversation_text = ""
                for msg in history:
                    role = "User" if msg.get('role') == 'user' else "Bot"
                    conversation_text += f"{role}: {msg.get('content', '')}\n"
                
                # Call SalesIQ API to create chat session
                api_result = salesiq_api.create_chat_session(session_id, conversation_text)
                logger.info(f"[SalesIQ] API result: {api_result}")
                
                # SalesIQ only supports "action": "reply" - transfer happens via API
                # Clear conversation after transfer
                if session_id in conversations:
                    del conversations[session_id]
                
                return {
                    "action": "reply",
                    "replies": ["I'm connecting you with our support team. If the transfer doesn't happen automatically, please call 1-888-415-5240 or email support@acecloudhosting.com for immediate assistance."],
                    "session_id": session_id
                }
        
        # Check for issue resolution
        resolution_keywords = ["resolved", "fixed", "working now", "solved", "all set"]
        if any(keyword in message_lower for keyword in resolution_keywords):
            logger.info(f"[SalesIQ] Issue resolved by user")
            response_text = "Great! I'm glad the issue is resolved. If you need anything else, feel free to ask!"
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})
            
            # Close chat in SalesIQ since issue is resolved
            close_result = salesiq_api.close_chat(session_id, "resolved")
            logger.info(f"[SalesIQ] Chat closure result: {close_result}")
            
            if session_id in conversations:
                del conversations[session_id]
            return {
                "action": "reply",
                "replies": [response_text],
                "session_id": session_id
            }
        
        # Check for not resolved
        not_resolved_keywords = ["not resolved", "not fixed", "not working", "didn't work", "still not", "still stuck"]
        if any(keyword in message_lower for keyword in not_resolved_keywords):
            logger.info(f"[SalesIQ] Issue NOT resolved - offering 3 options with interactive buttons")
            response_text = "I understand this is frustrating. Here are 3 ways I can help:"
            
            # Add to history so next response can find it
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})
            
            return {
                "action": "reply",
                "replies": [response_text],
                "suggestions": [
                    {
                        "text": "📞 Instant Chat",
                        "action_type": "reply",
                        "action_value": "1"
                    },
                    {
                        "text": "📅 Schedule Callback",
                        "action_type": "reply",
                        "action_value": "2"
                    },
                    {
                        "text": "🎫 Create Ticket",
                        "action_type": "reply",
                        "action_value": "3"
                    }
                ],
                "session_id": session_id
            }
        
        # Check for password reset - improved flow
        password_keywords = ["password", "reset", "forgot", "locked out"]
        if any(keyword in message_lower for keyword in password_keywords):
            logger.info(f"[SalesIQ] Password reset detected")
            # Check if user already answered about SelfCare registration
            if len(history) > 0:
                last_bot_message = history[-1].get('content', '') if history[-1].get('role') == 'assistant' else ''
                # If bot already asked about SelfCare registration
                if 'registered on the selfcare portal' in last_bot_message.lower():
                    # User is responding to that question
                    if 'yes' in message_lower or 'registered' in message_lower:
                        logger.info(f"[SalesIQ] User is registered on SelfCare")
                        response_text = "Great! Visit https://selfcare.acecloudhosting.com and click 'Forgot your password'. Let me know when you're there!"
                        conversations[session_id].append({"role": "user", "content": message_text})
                        conversations[session_id].append({"role": "assistant", "content": response_text})
                        return {
                            "action": "reply",
                            "replies": [response_text],
                            "session_id": session_id
                        }
                    elif 'no' in message_lower or 'not registered' in message_lower:
                        logger.info(f"[SalesIQ] User is NOT registered on SelfCare")
                        response_text = "No problem! For server/user account password reset, please contact our support team at 1-888-415-5240. They'll help you right away!"
                        conversations[session_id].append({"role": "user", "content": message_text})
                        conversations[session_id].append({"role": "assistant", "content": response_text})
                        return {
                            "action": "reply",
                            "replies": [response_text],
                            "session_id": session_id
                        }
            else:
                # First time asking about password reset
                logger.info(f"[SalesIQ] First password reset question - asking about SelfCare registration")
                response_text = "I can help! Are you registered on the SelfCare portal?"
                conversations[session_id].append({"role": "user", "content": message_text})
                conversations[session_id].append({"role": "assistant", "content": response_text})
                return {
                    "action": "reply",
                    "replies": [response_text],
                    "session_id": session_id
                }
        
        # Check for application updates
        app_update_keywords = ["update", "upgrade", "requires update", "needs update"]
        app_names = ["quickbooks", "lacerte", "drake", "proseries", "qb"]
        is_app_update = False
        if any(keyword in message_lower for keyword in app_update_keywords):
            if any(app in message_lower for app in app_names):
                is_app_update = True
        
        if is_app_update:
            logger.info(f"[SalesIQ] Application update request detected")
            response_text = "Application updates need to be handled by our support team to avoid downtime. Please contact support at:\n\nPhone: 1-888-415-5240 (24/7)\nEmail: support@acecloudhosting.com\n\nThey'll schedule the update for you!"
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})
            return {
                "action": "reply",
                "replies": [response_text],
                "session_id": session_id
            }
        
        # Check for option selections - INSTANT CHAT
        if "instant chat" in message_lower or "option 1" in message_lower or message_lower == "1" or "chat/transfer" in message_lower or payload == "option_1":
            logger.info(f"[SalesIQ] User selected: Instant Chat Transfer")
            
            try:
                # Build conversation history for agent to see
                conversation_text = ""
                for msg in history:
                    role = "User" if msg.get('role') == 'user' else "Bot"
                    conversation_text += f"{role}: {msg.get('content', '')}\n"
                
                # Prepare overrides from webhook payload
                req_meta = request.get('request', {}) if isinstance(request, dict) else {}
                override_app_id = req_meta.get('app_id') or getattr(salesiq_api, 'app_id', None)
                override_department_id = visitor.get('department_id') if isinstance(visitor, dict) else None
                
                # Extract visitor email as unique identifier (more reliable than IDs)
                visitor_email = visitor.get('email', 'support@acecloudhosting.com') if isinstance(visitor, dict) else 'support@acecloudhosting.com'
                
                # Call SalesIQ API (Visitor API) to create conversation and route to agent
                logger.info(f"[SalesIQ] Calling create_chat_session API with overrides app_id={override_app_id}, dept={override_department_id}, visitor_email={visitor_email}")
                
                # Pass visitor email as user_id (most reliable unique identifier per API docs)
                api_result = salesiq_api.create_chat_session(
                    visitor_email,  # Use email as unique user_id per API documentation
                    conversation_text,
                    app_id=override_app_id,
                    department_id=str(override_department_id) if override_department_id else None,
                    visitor_info=visitor
                )
                logger.info(f"[SalesIQ] API result: {api_result}")
            except Exception as api_error:
                logger.error(f"[SalesIQ] API call failed: {str(api_error)}")
                logger.error(f"[SalesIQ] Traceback: {traceback.format_exc()}")
            
            # SalesIQ webhooks only support "reply" action, not "transfer"
            # The transfer happens through the SalesIQ API call above
            # Send confirmation message to user
            response_text = "I'm connecting you with our support team. If the transfer doesn't happen automatically, please call 1-888-415-5240 or email support@acecloudhosting.com for immediate assistance."
            
            # Clear conversation after transfer
            if session_id in conversations:
                del conversations[session_id]
            
            return {
                "action": "reply",
                "replies": [response_text],
                "session_id": session_id
            }
        
        # Check for option selections - SCHEDULE CALLBACK
        if "callback" in message_lower or "option 2" in message_lower or message_lower == "2" or "schedule" in message_lower or payload == "option_2":
            logger.info(f"[SalesIQ] User selected: Schedule Callback")
            response_text = (
                "Perfect! I'm creating a callback request for you.\n\n"
                "Please provide:\n"
                "1. Your preferred time (e.g., 'tomorrow at 2 PM' or 'Monday morning')\n"
                "2. Your phone number\n\n"
                "Our support team will call you back at that time. A ticket has been created and you'll receive a confirmation email shortly.\n\n"
                "Thank you for contacting Ace Cloud Hosting!"
            )
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})

            # Fire-and-forget: protect external calls so webhook never breaks
            try:
                api_result = desk_api.create_callback_ticket(
                    user_email="support@acecloudhosting.com",
                    phone="pending",
                    preferred_time="pending",
                    issue_summary="Callback request from chat"
                )
                logger.info(f"[Desk] Callback ticket result: {api_result}")
            except Exception as e:
                logger.error(f"[Desk] Callback ticket error: {str(e)}")

            try:
                close_result = salesiq_api.close_chat(session_id, "callback_scheduled")
                logger.info(f"[SalesIQ] Chat closure result: {close_result}")
            except Exception as e:
                logger.error(f"[SalesIQ] Chat closure error: {str(e)}")

            # Clear conversation after callback (auto-close)
            if session_id in conversations:
                del conversations[session_id]

            return {
                "action": "reply",
                "replies": [response_text],
                "session_id": session_id
            }
        
        # Check for option selections - CREATE TICKET
        if "ticket" in message_lower or "option 3" in message_lower or message_lower == "3" or "support ticket" in message_lower or payload == "option_3":
            logger.info(f"[SalesIQ] User selected: Create Support Ticket")
            response_text = """Perfect! I'm creating a support ticket for you.

Please provide:
1. Your name
2. Your email
3. Your phone number
4. Brief description of the issue

A ticket will be created and you'll receive a confirmation email shortly. Our support team will follow up with you within 24 hours.

Thank you for contacting Ace Cloud Hosting!"""
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})
            
            # Call Desk API to create support ticket
            api_result = desk_api.create_support_ticket(
                user_name="pending",
                user_email="pending",
                phone="pending",
                description="Support ticket from chat",
                issue_type="general",
                conversation_history="\n".join([f"{msg.get('role')}: {msg.get('content')}" for msg in history])
            )
            logger.info(f"[Desk] Support ticket result: {api_result}")
            
            # Close chat in SalesIQ
            close_result = salesiq_api.close_chat(session_id, "ticket_created")
            logger.info(f"[SalesIQ] Chat closure result: {close_result}")
            
            # Clear conversation after ticket creation (auto-close)
            if session_id in conversations:
                del conversations[session_id]
            
            return {
                "action": "reply",
                "replies": [response_text],
                "session_id": session_id
            }
        #check for new request
        # Check for agent connection requests (legacy)
        agent_request_phrases = ["connect me to agent", "connect to agent", "human agent", "talk to human", "speak to agent"]
        if any(phrase in message_lower for phrase in agent_request_phrases):
            logger.info(f"[SalesIQ] User requesting human agent - offering options with interactive buttons")
            response_text = "I can help you with that. Here are your options:"
            
            conversations[session_id].append({"role": "user", "content": message_text})
            conversations[session_id].append({"role": "assistant", "content": response_text})
            
            return {
                "action": "reply",
                "replies": [response_text],
                "suggestions": [
                    {
                        "text": "📞 Instant Chat",
                        "action_type": "reply",
                        "action_value": "1"
                    },
                    {
                        "text": "📅 Schedule Callback",
                        "action_type": "reply",
                        "action_value": "2"
                    },
                    {
                        "text": "🎫 Create Ticket",
                        "action_type": "reply",
                        "action_value": "3"
                    }
                ],
                "session_id": session_id
            }
        
        # Check for acknowledgments - BUT NOT during step-by-step troubleshooting
        def is_acknowledgment_message(msg):
            msg = msg.lower().strip()
            # If message contains "then", it's likely a continuation, not an acknowledgment
            if 'then' in msg:
                return False
            # Only treat EXACT matches as acknowledgments (not partial)
            direct_acks = ["okay", "ok", "thanks", "thank you", "got it", "understood", "alright"]
            if msg in direct_acks:
                return True
            # Thanks patterns
            thanks_patterns = ["thank", "thnk", "thx", "ty"]
            if any(pattern in msg for pattern in thanks_patterns) and len(msg) < 20:
                return True
            return False
        
        # Check if we're in the middle of step-by-step troubleshooting
        is_in_troubleshooting = False
        if len(history) > 0:
            last_bot_message = history[-1].get('content', '') if history[-1].get('role') == 'assistant' else ''
            # Check for step-by-step guidance patterns
            troubleshooting_patterns = [
                'step',
                'can you',
                'do that',
                'let me know when',
                'can you see',
                'do you see',
                'click',
                'right-click',
                'press',
                'open',
                'navigate',
                'select',
                'find',
                'go to'
            ]
            if any(pattern in last_bot_message.lower() for pattern in troubleshooting_patterns):
                is_in_troubleshooting = True
        
        is_acknowledgment = is_acknowledgment_message(message_lower)
        
        if is_acknowledgment and not is_in_troubleshooting:
            logger.info(f"[SalesIQ] Acknowledgment detected (not in troubleshooting)")
            if message_lower in ["ok", "okay"]:
                logger.info(f"[SalesIQ] 'Ok/Okay' alone, asking if need more help")
                return {
                    "action": "reply",
                    "replies": ["Is there anything else I can help you with?"],
                    "session_id": session_id
                }
            else:
                logger.info(f"[SalesIQ] Acknowledgment with thanks detected")
                return {
                    "action": "reply",
                    "replies": ["You're welcome! Is there anything else I can help you with?"],
                    "session_id": session_id
                }
        elif is_acknowledgment and is_in_troubleshooting:
            logger.info(f"[SalesIQ] Acknowledgment during troubleshooting - continuing with LLM")
            # Fall through to LLM to continue with next step
        
        # Generate LLM response with embedded resolution steps
        logger.info(f"[SalesIQ] Calling OpenAI LLM with embedded resolution steps...")
        response_text = generate_response(message_text, history)
        
        # Clean response
        response_text = response_text.replace('**', '')
        import re
        response_text = re.sub(r'^\s*\*\s+', '- ', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'\n\s*\n+', '\n', response_text)
        response_text = response_text.strip()
        
        logger.info(f"[SalesIQ] Response generated: {response_text[:100]}...")
        
        # Update conversation history
        conversations[session_id].append({"role": "user", "content": message_text})
        conversations[session_id].append({"role": "assistant", "content": response_text})
        
        return {
            "action": "reply",
            "replies": [response_text],
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"[SalesIQ] ERROR: {str(e)}")
        logger.error(f"[SalesIQ] Traceback: {traceback.format_exc()}")
        return {
            "action": "reply",
            "replies": ["I'm having technical difficulties. Please call our support team at 1-888-415-5240."],
            "session_id": session_id or 'unknown'
        }

@app.post("/chat")
async def chat(request: ChatRequest):
    """Main chat endpoint for n8n webhook"""
    try:
        session_id = request.session_id
        message = request.message
        
        if session_id not in conversations:
            conversations[session_id] = []
        
        history = conversations[session_id]
        response_text = generate_response(message, history)
        
        conversations[session_id].append({"role": "user", "content": message})
        conversations[session_id].append({"role": "assistant", "content": response_text})
        
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset/{session_id}")
async def reset_conversation(session_id: str):
    """Reset conversation for a session"""
    if session_id in conversations:
        del conversations[session_id]
        return {"status": "success", "message": f"Conversation {session_id} reset"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    return {
        "active_sessions": len(conversations),
        "sessions": list(conversations.keys())
    }

# -----------------------------------------------------------
# Test endpoints to validate SalesIQ Visitor API transfer
# -----------------------------------------------------------
@app.get("/test/salesiq-transfer")
async def test_salesiq_transfer_get():
    """Quick GET test to exercise Visitor API with env defaults.
    
    IMPORTANT: Cannot use bot preview IDs (botpreview_...).
    This endpoint uses a real-looking email-based user ID for testing.
    """
    try:
        # Use email as user_id (most reliable per API docs) instead of session ID
        test_user_id = "vishal.dharan@acecloudhosting.com"
        conversation_text = "Test transfer from GET endpoint"
        logger.info(f"[Test] Initiating SalesIQ Visitor API transfer (GET) with user_id={test_user_id}")
        result = salesiq_api.create_chat_session(test_user_id, conversation_text)
        return {
            "user_id": test_user_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"[Test] SalesIQ transfer GET failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/salesiq-transfer")
async def test_salesiq_transfer_post(payload: Dict):
    """POST test to exercise Visitor API with overrides from payload.
    
    Accepts:
    - visitor_user_id: Unique identifier for visitor (use email, not botpreview_...)
    - conversation: Conversation text for agent
    - app_id: Override app_id
    - department_id: Override department_id
    - visitor: Full visitor info dict
    - custom_wait_time: Custom wait time
    
    IMPORTANT: visitor_user_id cannot be botpreview_... IDs.
    Use real email addresses or unique identifiers.
    """
    try:
        # Use email as user_id (more reliable than session IDs)
        visitor_user_id = str(payload.get("visitor_user_id") or "vishal.dharan@acecloudhosting.com")
        conversation_text = str(payload.get("conversation") or "Test transfer from POST endpoint")
        app_id = payload.get("app_id")
        department_id = payload.get("department_id")
        visitor_info = payload.get("visitor")
        custom_wait_time = payload.get("custom_wait_time")

        logger.info(
            f"[Test] Initiating SalesIQ Visitor API transfer (POST) for user_id={visitor_user_id} with app_id={app_id}, dept={department_id}"
        )
        result = salesiq_api.create_chat_session(
            visitor_user_id,  # Use as unique user_id per API documentation
            conversation_text,
            app_id=app_id,
            department_id=str(department_id) if department_id is not None else None,
            visitor_info=visitor_info,
            custom_wait_time=custom_wait_time,
        )
        return {
            "user_id": visitor_user_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"[Test] SalesIQ transfer POST failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    print("="*70)
    print("ACE CLOUD HOSTING - SUPPORT BOT")
    print("="*70)
    print(f"\n[STARTING] FastAPI server on port {port}...")
    print(f"[ENDPOINT] http://0.0.0.0:{port}")
    print(f"[DOCS] http://0.0.0.0:{port}/docs")
    print("\n[READY] Ready to receive webhooks!")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
