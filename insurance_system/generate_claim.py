import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)


def create_rag_dataset_report(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=LETTER,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = []
    styles = getSampleStyleSheet()

    # --- RAG-Optimized Styles ---
    # High contrast headers for better chunking
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Heading1"],
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=24,
            spaceBefore=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            parent=styles["Heading2"],
            fontSize=16,
            textColor=colors.darkblue,
            spaceBefore=20,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubHeader",
            parent=styles["Heading3"],
            fontSize=12,
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        )
    )

    # BodyText usually exists in sample stylesheet, update it or add if missing
    if "BodyText" in styles:
        styles["BodyText"].fontSize = 10.5
        styles["BodyText"].leading = 14
        styles["BodyText"].alignment = TA_JUSTIFY
    else:
        styles.add(
            ParagraphStyle(
                name="BodyText",
                parent=styles["Normal"],
                fontSize=10.5,
                leading=14,
                alignment=TA_JUSTIFY,
            )
        )
    styles.add(
        ParagraphStyle(
            name="DataPoint",
            parent=styles["Normal"],
            fontSize=10,
            fontName="Courier",
            leading=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableText", parent=styles["Normal"], fontSize=9, leading=11
        )
    )

    # Define Quote style if missing
    if "Quote" not in styles:
        styles.add(
            ParagraphStyle(
                name="Quote",
                parent=styles["Normal"],
                leftIndent=20,
                rightIndent=20,
                fontName="Helvetica-Oblique",
                fontSize=10,
            )
        )

    # Helper to add standard separator
    def add_separator():
        story.append(Spacer(1, 12))
        story.append(Paragraph("_" * 85, styles["BodyText"]))
        story.append(Spacer(1, 12))

    # =============================================================================================
    # PAGE 1: CLAIM SUMMARY & METADATA (Structured for Entity Extraction)
    # =============================================================================================

    story.append(Paragraph("PROPERTY LOSS COMPREHENSIVE REPORT", styles["ReportTitle"]))
    story.append(
        Paragraph("<b>CONFIDENTIAL CLAIM FILE: HO-2024-8892</b>", styles["Title"])
    )
    story.append(Spacer(1, 20))

    # DATA ENRICHMENT: Structured Metadata Table
    meta_data = [
        ["FIELD", "VALUE", "CONFIDENCE"],
        ["Claim ID", "HO-2024-8892", "High"],
        ["Policy Number", "POL-TX-99824-HO3", "Verified"],
        ["Primary Insured", "Alex Johnson", "Verified"],
        ["Risk Address", "124 Maple Street, Austin, TX 78701", "Verified"],
        ["Date of Loss", "November 16, 2024", "Confirmed"],
        ["Cause of Loss", "Sudden & Accidental Discharge (Water)", "Confirmed"],
        ["Adjuster", "Mike Ross (License #TX-44921)", "Active"],
        ["Total Payout", "$19,550.00", "Final"],
        ["Status", "CLOSED - PAYMENT ISSUED", "Final"],
    ]
    t = Table(meta_data, colWidths=[2 * inch, 3.5 * inch, 1.5 * inch])
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 20))

    story.append(Paragraph("1.0 EXECUTIVE OVERVIEW", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "On November 16, 2024, the insured property sustained significant water damage due to a plumbing failure in the second-floor master bathroom. The loss was captured by smart home telemetry, specifically a 'WaterHero' flow monitor and Google Nest ecosystem. The incident resulted in saturation of the second-floor vanity, subfloor, and first-floor living room ceiling and hardwood flooring. Emergency mitigation was performed by DryFast Inc., followed by restoration by Austin Home Restorations LLC. All coverage determinations have been finalized under Policy Form HO-3.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 2: POLICY LANGUAGE & COVERAGE ANALYSIS (For Semantic Search)
    # =============================================================================================

    story.append(Paragraph("2.0 POLICY CONTRACT ANALYSIS", styles["SectionHeader"]))
    story.append(Paragraph("<b>2.1 Applicable Policy Forms</b>", styles["SubHeader"]))
    story.append(
        Paragraph(
            "The coverage is adjudicated under the standard HO-3 'Special Form' Homeowners Policy (Ed. 05/11). The following sections are relevant to the coverage determination:",
            styles["BodyText"],
        )
    )

    # DATA ENRICHMENT: Synthetic Policy Clauses
    clauses = [
        ["Section", "Provision", "Application to Claim"],
        [
            "Section I",
            "Coverage A - Dwelling",
            "Applies to structural damage (floors, ceiling, vanity). Limit: $450,000.",
        ],
        [
            "Section I",
            "Coverage C - Personal Property",
            "Applies to contents (Rug, TV, Sofa). Limit: $225,000 (50% of Cov A).",
        ],
        [
            "Section I",
            "Perils Insured Against",
            "Covers 'Sudden and accidental discharge or overflow of water' from a plumbing system.",
        ],
        [
            "Exclusions",
            "Mold/Fungus",
            "Limited coverage ($5,000) applies if mold results from a covered water loss. (Not triggered in this claim due to rapid mitigation).",
        ],
        [
            "Conditions",
            "Duties After Loss",
            "Insured complied by contacting mitigation services immediately (DryFast Inc).",
        ],
    ]
    t_pol = Table(clauses, colWidths=[1.2 * inch, 2 * inch, 4 * inch])
    t_pol.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(t_pol)

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>2.2 Deductible Logic</b>", styles["SubHeader"]))
    story.append(
        Paragraph(
            "The policy carries a $1,000.00 All-Peril deductible. This deductible is applied once per occurrence. In this calculation, the deductible was subtracted from the Coverage A (Dwelling) payment, resulting in a net dwelling payment of $11,400.00.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 3: IOT FORENSICS & SENSOR DATA (Time-Series Data)
    # =============================================================================================

    story.append(Paragraph("3.0 IOT FORENSIC DATA ANALYSIS", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "The following data was extracted from the insured's 'WaterHero' and 'Google Nest' APIs. This objective data serves as the primary verification of the loss timeline.",
            styles["BodyText"],
        )
    )

    # DATA ENRICHMENT: Detailed Time Series
    story.append(
        Paragraph(
            "<b>3.1 High-Resolution Sensor Log (11/16/2024)</b>", styles["SubHeader"]
        )
    )

    sensor_data = [
        ["TIMESTAMP (CST)", "DEVICE ID", "METRIC", "VALUE", "STATE CHANGE"],
        ["10:20:00 AM", "Flow_Meter_01", "Flow Rate", "0.0 GPM", "Normal"],
        ["10:22:15 AM", "Flow_Meter_01", "Flow Rate", "8.5 GPM", "ABNORMAL START"],
        ["10:22:20 AM", "Hum_Sensor_MB", "Rel. Humidity", "48%", "Rising (+3%)"],
        ["10:25:00 AM", "Flow_Meter_01", "Total Vol", "25.5 Gal", "Continuous"],
        ["10:45:00 AM", "Cam_LvRm_02", "Audio Level", "45 dB", "Dripping Detected"],
        ["11:00:00 AM", "Hum_Sensor_MB", "Rel. Humidity", "85%", "Saturation Alert"],
        ["11:15:00 AM", "Flow_Meter_01", "Total Vol", "448.5 Gal", "CRITICAL ALERT"],
        ["11:15:05 AM", "Valve_Ctrl_Main", "Valve State", "CLOSED", "AUTO-SHUTOFF"],
        ["11:15:10 AM", "Flow_Meter_01", "Flow Rate", "0.0 GPM", "Leak Stopped"],
    ]
    t_sensor = Table(
        sensor_data,
        colWidths=[1.5 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 2 * inch],
    )
    t_sensor.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Courier-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Courier"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (4, 2), (4, 2), colors.red),  # Highlight abnormal start
                ("TEXTCOLOR", (4, 8), (4, 8), colors.green),  # Highlight shutoff
            ]
        )
    )
    story.append(t_sensor)

    story.append(
        Paragraph("<b>3.2 Physics of Failure Analysis</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "The flow rate of 8.5 GPM is consistent with a complete severance of a 3/8-inch compression supply line under standard municipal pressure (60-80 PSI). The delay between the leak start (10:22 AM) and the auto-shutoff (11:15 AM) indicates the 'Smart Home' protocol was set to a '45-minute continuous flow' threshold before triggering. While this setting prevented immediate shutoff, it successfully prevented catastrophic flooding that would have occurred over the subsequent 3 hours before the insured returned.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 4: MITIGATION LOGS (Psychrometric Data)
    # =============================================================================================

    story.append(Paragraph("4.0 MITIGATION & DRYING PROTOCOL", styles["SectionHeader"]))
    story.append(
        Paragraph(
            "DryFast Inc. (Vendor ID: V-9982) mobilized immediately. The following psychrometric logs document the drying process of the structural materials.",
            styles["BodyText"],
        )
    )

    # DATA ENRICHMENT: Equipment List
    story.append(Paragraph("<b>4.1 Equipment Deployment</b>", styles["SubHeader"]))
    story.append(
        Paragraph(
            "• 6x Phoenix AirMax Radial Air Movers (Assets #AM-101 to AM-106)",
            styles["DataPoint"],
        )
    )
    story.append(
        Paragraph(
            "• 1x Drieaz Evolution LGR Dehumidifier (Asset #DH-22)", styles["DataPoint"]
        )
    )
    story.append(
        Paragraph("• 1x HEPA 500 Air Scrubber (Asset #AS-09)", styles["DataPoint"])
    )

    # DATA ENRICHMENT: Drying Log Table
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<b>4.2 Daily Psychrometric Log (Drying Progress)</b>", styles["SubHeader"]
        )
    )
    drying_data = [
        [
            "Date",
            "Time",
            "Amb Temp",
            "Amb RH",
            "GPP",
            "Material",
            "Moisture Content",
            "Status",
        ],
        [
            "11/16/24",
            "17:00",
            "72°F",
            "65%",
            "78",
            "Oak Floor",
            "99.9% (WME)",
            "SATURATED",
        ],
        [
            "11/17/24",
            "09:00",
            "85°F",
            "35%",
            "62",
            "Oak Floor",
            "45.0% (WME)",
            "Drying",
        ],
        [
            "11/18/24",
            "09:00",
            "85°F",
            "30%",
            "52",
            "Oak Floor",
            "22.0% (WME)",
            "Drying",
        ],
        [
            "11/19/24",
            "09:00",
            "82°F",
            "28%",
            "45",
            "Oak Floor",
            "12.0% (WME)",
            "Dry Standard Met",
        ],
    ]
    t_dry = Table(
        drying_data,
        colWidths=[
            0.8 * inch,
            0.6 * inch,
            0.8 * inch,
            0.8 * inch,
            0.6 * inch,
            1 * inch,
            1.2 * inch,
            1.2 * inch,
        ],
    )
    t_dry.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_dry)

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<b>Note:</b> While the moisture content was reduced to 12%, the physical structure of the white oak boards exhibited permanent 'cupping' (curvature of the wood), necessitating replacement despite successful drying.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 5: DETAILED REPAIR ESTIMATE (Xactimate Style)
    # =============================================================================================

    story.append(
        Paragraph(
            "5.0 DETAILED REPAIR ESTIMATE (SCOPE OF WORK)", styles["SectionHeader"]
        )
    )
    story.append(
        Paragraph("<b>Vendor:</b> Austin Home Restorations LLC", styles["DataPoint"])
    )
    story.append(
        Paragraph("<b>Estimate ID:</b> ES-2024-8892-REV2", styles["DataPoint"])
    )
    story.append(Spacer(1, 10))

    # DATA ENRICHMENT: Detailed Line Items
    est_data = [
        ["CAT", "DESCRIPTION", "QTY", "UNIT", "UNIT PRICE", "TOTAL"],
        ["<b>LIV</b>", "<b>Living Room (Downstairs)</b>", "", "", "", ""],
        ["WTR", "Tear out wet drywall, cleanup, bag", "16.00", "SF", "$2.55", "$40.80"],
        ["DRY", 'Install 5/8" drywall - hung, taped', "16.00", "SF", "$3.45", "$55.20"],
        [
            "PNT",
            "Seal/Prime/Paint ceiling (2 coats)",
            "400.00",
            "SF",
            "$1.10",
            "$440.00",
        ],
        [
            "FCW",
            "Remove hardwood flooring (White Oak)",
            "400.00",
            "SF",
            "$3.00",
            "$1,200.00",
        ],
        [
            "FCW",
            "Install White Oak plank flooring",
            "400.00",
            "SF",
            "$12.00",
            "$4,800.00",
        ],
        ["FCW", "Sand, stain, and finish floor", "400.00", "SF", "$4.00", "$1,600.00"],
        ["<b>BTH</b>", "<b>Master Bath (Upstairs)</b>", "", "", "", ""],
        ["CAB", "Detach & Reset Vanity Top", "1.00", "EA", "$250.00", "$250.00"],
        [
            "CAB",
            "Vanity - High Grade - Replace",
            "1.00",
            "EA",
            "$1,500.00",
            "$1,500.00",
        ],
        ["PLM", "P-Trap/Supply Line assembly", "1.00", "EA", "$350.00", "$350.00"],
        ["PLM", "Plumber Labor - Emergency Call", "1.00", "EA", "$500.00", "$500.00"],
        ["<b>GEN</b>", "<b>General / Overhead</b>", "", "", "", ""],
        [
            "MAT",
            "Construction Materials / Waste",
            "1.00",
            "EA",
            "$1,664.00",
            "$1,664.00",
        ],
        ["", "<b>TOTAL ESTIMATE</b>", "", "", "", "<b>$12,400.00</b>"],
    ]
    t_est = Table(
        est_data,
        colWidths=[0.5 * inch, 3 * inch, 0.8 * inch, 0.6 * inch, 1 * inch, 1.2 * inch],
    )
    t_est.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                ("LINEABOVE", (-1, -1), (-1, -1), 1, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 1), (-1, 1), colors.whitesmoke),  # LIV header
                ("BACKGROUND", (0, 8), (-1, 8), colors.whitesmoke),  # BTH header
                ("BACKGROUND", (0, 13), (-1, 13), colors.whitesmoke),  # GEN header
            ]
        )
    )
    story.append(t_est)

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 6: CONTENTS INVENTORY & VALUATION (Methodology)
    # =============================================================================================

    story.append(
        Paragraph("6.0 PERSONAL PROPERTY VALUATION (RCV/ACV)", styles["SectionHeader"])
    )
    story.append(
        Paragraph(
            "The following items were claimed under Coverage C. Valuation was determined using 'Actual Cash Value' (ACV) vs 'Replacement Cost Value' (RCV) analysis.",
            styles["BodyText"],
        )
    )

    # DATA ENRICHMENT: Valuation Analysis
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>6.1 Persian Rug (Total Loss)</b>", styles["SubHeader"]))
    story.append(
        Paragraph(
            "• <b>Original Purchase:</b> $3,500 (Rugs Direct, 2019).<br/>• <b>Damage Assessment:</b> Irreversible dye migration (bleeding) due to saturation. Wool pile degradation.<br/>• <b>Market Analysis:</b> Comparable hand-knotted Heriz rugs (8x10) currently retail for $3,400 - $3,800. <br/>• <b>Decision:</b> Full Policy Limit payout of $3,500 approved.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 10))
    story.append(
        Paragraph("<b>6.2 Samsung QLED TV (Electronics)</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "• <b>Diagnostics:</b> Unit failed power-on self-test (POST). Moisture detected on PCB.<br/>• <b>Replacement Model:</b> Samsung QN90C Series (Current Equivalent).<br/>• <b>Cost:</b> $1,200.00 (verified via Best Buy current pricing).<br/>• <b>Depreciation:</b> Waived under Replacement Cost coverage endorsement.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 10))
    story.append(
        Paragraph("<b>6.3 West Elm Sofa (Partial Loss)</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "• <b>Claim:</b> Water spots and odor.<br/>• <b>Assessment:</b> Leather is not porous enough to absorb deep water in 50 minutes. Damage is cosmetic.<br/>• <b>Remedy:</b> Professional leather cleaning and conditioning.<br/>• <b>Allowance:</b> $250.00 (Accepted by insured).",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 7: PLUMBING & CAUSATION REPORT (Technical)
    # =============================================================================================

    story.append(
        Paragraph("7.0 CAUSATION & TECHNICAL PLUMBING REPORT", styles["SectionHeader"])
    )

    story.append(
        Paragraph("<b>7.1 Component Failure Analysis</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "The failure occurred at the supply line connection to the 'cold' side of the master bathroom vanity faucet. The specific component identified is a braided stainless steel supply line with a 3/8-inch compression nut.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            "<b>Failure Mode:</b> Stress Corrosion Cracking (SCC) vs Mechanical Failure. <br/>Based on the text evidence from the repairing plumber ('Joe'), the diagnosis was a 'faulty compression nut'. This suggests the nut sheared off the threads, likely due to overtightening during initial installation or a manufacturing defect in the metallurgy of the nut.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph("<b>7.2 Plumber Statement (Reconstructed)</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            '<b>Provider:</b> Joe the Plumber (Lic #4482)<br/><b>Date:</b> Nov 16, 2024<br/><b>Statement:</b> "Arrived 3:45 PM. Found supply line disconnected. The nut had split vertically. No evidence of corrosion or slow leak prior to burst. Replaced with new fluidmaster line. Verified leak free."',
            styles["Quote"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 8: SUBROGATION & LEGAL ANALYSIS
    # =============================================================================================

    story.append(
        Paragraph("8.0 SUBROGATION & RECOVERY ANALYSIS", styles["SectionHeader"])
    )
    story.append(
        Paragraph(
            "Subrogation is the process of recovering costs from a liable third party. An analysis was conducted to determine if the loss could be tendered to the supply line manufacturer or the installer.",
            styles["BodyText"],
        )
    )

    # DATA ENRICHMENT: Legal Logic
    story.append(
        Paragraph("<b>8.1 Target 1: Supply Line Manufacturer</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "<b>Potential Liability:</b> Product Defect.<br/><b>Analysis:</b> The supply line is approximately 5 years old (based on home age). To prove a manufacturing defect, the failed nut would need to be retained and sent to a metallurgy lab for destructive testing (Cost ~$2,500).<br/><b>Decision:</b> <u>Cost Prohibitive.</u> The claim value ($19k) does not justify the forensic expense.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph(
            "<b>8.2 Target 2: Smart Valve Manufacturer (WaterHero)</b>",
            styles["SubHeader"],
        )
    )
    story.append(
        Paragraph(
            "<b>Potential Liability:</b> Failure to prevent loss.<br/><b>Analysis:</b> Review of the Terms of Service (ToS) for the device indicates it is a 'monitoring aid' and does not guarantee leak prevention. Furthermore, the logs show the device <i>did</i> activate according to its 'Away' profile settings (45 min delay).<br/><b>Decision:</b> <u>No Liability.</u> The device functioned as programmed.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph("<b>8.3 Target 3: Installing Plumber</b>", styles["SubHeader"])
    )
    story.append(
        Paragraph(
            "<b>Potential Liability:</b> Improper installation (overtightening).<br/><b>Analysis:</b> Statue of Repose in Texas for improvements to real property is 10 years. However, proving overtightening 5 years post-installation is factually difficult without the original installer's records.<br/><b>Decision:</b> <u>Close without Subrogation.</u>",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 9: COMMUNICATION LOGS (Expanded)
    # =============================================================================================

    story.append(Paragraph("9.0 COMMUNICATION & CONTACT LOG", styles["SectionHeader"]))

    # DATA ENRICHMENT: Full Log
    comm_data = [
        ["Date/Time", "Party", "Direction", "Summary"],
        [
            "11/16 14:45",
            "Insured",
            "Inbound",
            "FNOL Call. Reported 'house is a swimming pool'.",
        ],
        [
            "11/16 15:00",
            "Adjuster",
            "Outbound",
            "Auth for Mitigation (DryFast) given verbally.",
        ],
        [
            "11/18 09:00",
            "Adjuster",
            "Internal",
            "Reviewed Sensor Logs. Coverage confirmed.",
        ],
        [
            "11/18 11:00",
            "Adjuster",
            "Field",
            "On-Site Inspection. Photos taken. Measurments: 400sqft.",
        ],
        ["11/20 14:00", "DryFast", "Inbound", "Drying complete. Inv $3,500 received."],
        [
            "11/22 10:00",
            "Adjuster",
            "Outbound",
            "Settlement offer email sent. Rug = Total Loss.",
        ],
        [
            "11/22 10:45",
            "Insured",
            "Inbound",
            "Accepted settlement. Agreed to sofa cleaning.",
        ],
        ["11/24 09:00", "Finance", "Internal", "Payment issued via EFT. File Closed."],
    ]
    t_comm = Table(comm_data, colWidths=[1.2 * inch, 1 * inch, 1 * inch, 4 * inch])
    t_comm.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.append(t_comm)

    story.append(PageBreak())

    # =============================================================================================
    # PAGE 10: FINAL FINANCIAL RECONCILIATION
    # =============================================================================================

    story.append(
        Paragraph("10.0 FINAL FINANCIAL RECONCILIATION", styles["SectionHeader"])
    )
    story.append(
        Paragraph(
            "The following ledger represents the final disposition of funds for Claim HO-2024-8892.",
            styles["BodyText"],
        )
    )

    final_ledger = [
        ["CATEGORY", "PAYEE", "STATUS", "AMOUNT"],
        ["Emergency Mitigation", "DryFast Inc.", "Paid (Direct)", "$3,500.00"],
        ["Dwelling Repairs", "Alex Johnson", "Paid (Indemnity)", "$12,400.00"],
        ["  (Less Deductible)", "", "", "-$1,000.00"],
        ["Contents (Rug)", "Alex Johnson", "Paid", "$3,500.00"],
        ["Contents (TV)", "Alex Johnson", "Paid", "$1,200.00"],
        ["Contents (Laptop)", "Alex Johnson", "Paid", "$2,000.00"],
        ["Contents (Sofa Clean)", "Alex Johnson", "Paid", "$250.00"],
        ["ALE (Hotel)", "Marriott / A. Johnson", "Reimbursed", "$1,200.00"],
        ["", "", "<b>NET PAYOUT</b>", "<b>$19,550.00</b>"],
    ]
    t_fin = Table(final_ledger, colWidths=[2 * inch, 2 * inch, 1.5 * inch, 1.5 * inch])
    t_fin.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, -1), (-1, -1), colors.lightgrey),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story.append(t_fin)

    add_separator()

    story.append(Paragraph("<b>CERTIFICATION</b>", styles["SubHeader"]))
    story.append(
        Paragraph(
            "I certify that this report represents a true and accurate investigation of the facts and circumstances regarding the captioned loss. All payments have been issued in accordance with the policy provisions and applicable Texas Insurance Codes.",
            styles["BodyText"],
        )
    )

    story.append(Spacer(1, 40))
    story.append(Paragraph("Signed:", styles["BodyText"]))
    story.append(Paragraph("<b>Mike Ross</b>", styles["SubHeader"]))
    story.append(
        Paragraph("Senior Property Adjuster | SafeGuard Insurance", styles["BodyText"])
    )

    # Build PDF
    doc.build(story)
    print(f"RAG Dataset Generated: {filename}")


if __name__ == "__main__":
    create_rag_dataset_report("RAG_Claim_HO-2024-8892.pdf")
