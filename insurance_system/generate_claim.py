import os

from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 10)
        self.cell(0, 10, "CLAIM FILE: HO-2024-8892 | CONFIDENTIAL", 0, 1, "R")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def create_claim_pdf():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Define the pages content
    pages = [
        # PAGE 1
        """CLAIM SUMMARY COVER SHEET

CLAIM ID: HO-2024-8892
POLICY HOLDER: Alex Johnson
PROPERTY: 124 Maple Street, Austin, TX
STATUS: CLOSED (Payment Issued)

INCIDENT SUMMARY:
Date of Incident: Saturday, November 16, 2024
Cause: Burst pipe in second-floor bathroom vanity (Sudden & Accidental).
Description: Insured returned home to find water running. Smart valve had activated but significant damage occurred to upstairs bath and downstairs living room (ceiling, floors, contents).

FINANCIAL BREAKDOWN:
- Emergency Mitigation (DryFast Inc): $3,500.00 (Direct Pay)
- Dwelling Repairs (Contractor): $12,400.00
- Personal Property (Contents): $6,950.00
- Loss of Use (Hotel): $1,200.00
- Deductible: -$1,000.00

TOTAL NET PAYOUT TO INSURED: $19,550.00
""",
        # PAGE 2
        """INCIDENT REPORT (FNOL TRANSCRIPT)

Date: Nov 16, 2024 | Time: 2:45 PM
Caller: Alex Johnson (Insured)
Agent: Sarah (Support Hotline)

[TRANSCRIPT START]
Agent: SafeGuard Insurance, this is Sarah. How can I help?
Alex: Hi, I need to file a claim. I just got home and my house is a swimming pool.
Agent: Are you safe? Is the water stopped?
Alex: Yes, my smart water valve shut it off automatically, but not before it ruined everything. I walked in 10 minutes ago. Water is dripping from the living room ceiling lights. My Persian rug is soaked.
Agent: Do you know the source?
Alex: Upstairs master bath. The metal supply line under the sink just snapped off.
Agent: Okay, opening Claim HO-2024-8892. You have authorization to call a mitigation crew immediately.
[TRANSCRIPT END]
""",
        # PAGE 3
        """EXHIBIT A: SMARTHOME SENSOR LOGS
System: Google Nest / WaterHero Integration
Date: November 16, 2024

DATA GRANULARITY: SECOND/MINUTE INTERVALS

| Timestamp  | Device Name      | Event / Reading          | Interpretation           |
|------------|------------------|--------------------------|--------------------------|
| 09:00:00 AM| Front Door Lock  | Status: Locked           | Alex leaves house        |
| 10:00:00 AM| Master Bath Hum  | Reading: 45%             | Normal                   |
| 10:22:15 AM| Main Flow Meter  | Flow: 8.5 Gal/Min        | **LEAK STARTS** |
| 10:22:20 AM| Master Bath Hum  | Reading: 48%             | Humidity rising          |
| 10:30:00 AM| Main Flow Meter  | Flow: 8.4 Gal/Min        | Leak continuous          |
| 10:45:00 AM| Living Room Cam  | Audio: "Dripping"        | Water hits downstairs    |
| 11:00:00 AM| Master Bath Hum  | Reading: 85%             | Saturation               |
| 11:15:00 AM| Main Flow Meter  | Alert: Continuous Flow   | System Alert             |
| 11:15:05 AM| Smart Valve      | Action: Auto-Shutoff     | **VALVE CLOSES** |
| 11:15:10 AM| Main Flow Meter  | Flow: 0.0 Gal/Min        | **LEAK STOPPED** |
| 02:30:15 PM| Living Room Cam  | Motion Detected          | Alex returns home        |
""",
        # PAGE 4
        """TEXT MESSAGE HISTORY

PARTIES: Alex Johnson & "Joe the Plumber"
DATE: Nov 16, 2024

[02:35 PM] Alex: Joe, emergency. Pipe burst.
[02:38 PM] Joe: I can be there in 20 mins.
[02:39 PM] Alex: Under sink. Water everywhere. Auto-valve caught it but floors are ruined.
[03:45 PM] Joe: Fixed. It was a faulty compression nut. Invoice sent. $250.
[03:46 PM] Alex: Thanks.

PARTIES: Alex Johnson & "DryFast Inc" (Mitigation)
DATE: Nov 16, 2024

[03:00 PM] Alex: Insurance told me to call. Need extraction ASAP.
[05:00 PM] DryFast: Crew arrived.
[09:00 PM] DryFast: Fans setup. Do not unplug. We return Monday.
""",
        # PAGE 5
        """ADJUSTER DIARY (INTERNAL NOTES)
Adjuster: Mike Ross
Date: Nov 18, 2024

09:00 AM: Reviewed sensor logs (Page 3). Confirms sudden burst at 10:22 AM. Coverage verified.
11:00 AM: Site Inspection.
   - Master Bath: Vanity warped. Tile grout wet.
   - Living Room: Ceiling stain 4x4ft. Drywall sagging. Hardwood floors cupping.
   - Contents:
     1. Persian Rug: Soaked, bleeding colors. Total loss.
     2. Samsung TV: Wet on back panel. Needs testing.
     3. West Elm Sofa: Wet leather, smells. Recommend cleaning first.

02:00 PM: Authorized hotel stay (ALE) due to fan noise.
""",
        # PAGE 6
        """CONTENTS INVENTORY (CLAIMED ITEMS)
Submitted by: Alex Johnson

| Item | Description            | Age | Price    | Condition Claimed |
|------|------------------------|-----|----------|-------------------|
| 1    | Samsung 65" QLED TV    | 2yr | $1,200   | Won't turn on     |
| 2    | Persian Rug (8x10)     | 5yr | $3,500   | Soaked/Stained    |
| 3    | MacBook Pro            | 3yr | $2,000   | Screen damaged    |
| 4    | West Elm Leather Sofa  | 4yr | $1,800   | Water spots/Smell |
| 5    | Hardwood Flooring      | 10yr| Unknown  | Warped boards     |
""",
        # PAGE 7
        """PROOF OF PURCHASE / RECEIPTS

[RECEIPT 1: BEST BUY - 11/24/2022]
Item: Samsung 65" QLED 4K TV
Total: $1,298.99
Payment: Visa **** 4492

[RECEIPT 2: RUGS DIRECT - 05/15/2019]
Item: Authentic Heriz Wool Rug
Price: $3,500.00
Status: Delivered

[RECEIPT 3: APPLE STORE - 01/10/2021]
Item: MacBook Pro 13-inch
Total: $2,248.00
""",
        # PAGE 8
        """REPAIR ESTIMATE
Vendor: Austin Home Restorations LLC
Scope: Living Room & Bath Repair

LIVING ROOM:
- Demo hardwood (400sqft): $1,200
- Install White Oak (400sqft): $4,800
- Sand/Refinish: $1,600
- Drywall Ceiling Repair/Paint: $1,100

BATHROOM:
- Replace Vanity Cabinet: $1,500
- Plumbing Labor: $500

Materials: $1,700

TOTAL ESTIMATE: $12,400.00
""",
        # PAGE 9
        """EMAIL CORRESPONDENCE: SETTLEMENT
From: Mike Ross (Adjuster)
To: Alex Johnson
Date: Nov 22, 2024

Hi Alex,
Decision on your items:
1. Rug: Approved ($3,500) - Total loss.
2. TV/Laptop: Approved.
3. Sofa: PARTIALLY DENIED replacement. We are approving $250 for professional cleaning. If cleaning fails, we will revisit replacement.

Please confirm if you accept.
- Mike

---
From: Alex Johnson
To: Mike Ross
Date: Nov 22, 2024

Mike,
I accept the cleaning attempt for the sofa. If it still smells, I will let you know. I accept the rest.
- Alex
""",
        # PAGE 10
        """FINAL SETTLEMENT LETTER
Date: Nov 24, 2024

SUMMARY OF PAYMENTS ISSUED:

1. DWELLING (Repairs)
   Estimate: $12,400.00
   Less Deductible: -$1,000.00
   Net: $11,400.00

2. PERSONAL PROPERTY
   Rug: $3,500.00
   TV: $1,200.00
   Laptop: $2,000.00
   Sofa Cleaning: $250.00
   Net: $6,950.00

3. ALE (Hotel)
   Marriott (3 Nights): $1,200.00

TOTAL PAYMENT TRANSFERRED: $19,550.00
""",
    ]

    # Generate pages
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        # Use multi_cell for text wrapping
        pdf.multi_cell(0, 6, page_text)

    # Save to data directory
    output_path = os.path.join(
        os.path.dirname(__file__), "data", "claim_HO-2024-8892.pdf"
    )
    # Ensure data dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf.output(output_path)
    print(f"PDF generated successfully: {output_path}")


if __name__ == "__main__":
    create_claim_pdf()
