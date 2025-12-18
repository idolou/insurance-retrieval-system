# System Walkthrough & Verification

This document captures a live run of the Insurance Retrieval Agent, demonstrating both **Summary** and **Needle** (specific fact) retrieval capabilities as well as **MCP** (Multi-Context Processing) capabilities.

## 🖥️ Interactive CLI Session

Below is the transcript of the session running `python insurance_system/main.py`.

```text
╭───────────────────────────╮
│ Insurance Retrieval Agent │
╰───────────────────────────╯
⚙️ Initializing Insurance Claim Retrieval System...
✅ System Ready! Type 'exit' to quit.
Type '1' to see more sample queries.


👤 User > Summarize the claim.

🤖 Agent >
🛠️ Called Tool: summary_expert
{'__arg1': 'Summarize the claim.'}
   → Result (summary_expert):
The claim involved a property loss due to a plumbing failure that led to significant water damage in the
insured property. This incident occurred on November 16, 2024, and was detected by smart home devices.
Emergency services were quickly engaged for mitigation and restoration. The incident was covered under the
standard HO-3 'Special Form' Homeowners Policy with a $1,000 deductible. Forensic analysis confirmed the
event's timeline, with successful drying of structural materials despite permanent damage to some
flooring. Detailed repair estimates and valuations were conducted, resulting in a net payout of $19,550.00
to the insured. The plumbing failure was caused by stress corrosion cracking in the supply line, and
although subrogation analysis was performed, no recovery from third parties was pursued. Communication and
financial records documented the claim's progression.
--------------------------------------------------

👤 User > What is the deductible?

🤖 Agent >
🛠️ Called Tool: needle_expert
{'__arg1': 'deductible'}
   → Result (needle_expert):
$1,000.00
The deductible for the claim is $1,000.00.
--------------------------------------------------

👤 User > exit

Shutting down. Goodbye!
```

## ✅ Validation

- **Routing Logic**:
  - "Summarize the claim" -> Routed correctly to `summary_expert`.
  - "What is the deductible?" -> Routed correctly to `needle_expert`.
- **Retrieval Accuracy**:
  - Summary covers the core incident details (Plumbing failure, Nov 16, $19,550 payout).
  - Needle retrieval successfully found the exact deductible amount **$1,000.00**.

### 🧩 MCP Tool Integration (Weather & Time)

Below is an example of the system answering a complex claim-context question by using **Needs** (to find location/date) and an **MCP Tool** (to check historical weather).

**Query**: `"Find the location and date of the incident from the claim, then check the historical weather for that day to see if it contributed to the loss."`

```text
🤖 Agent >
🛠️ Called Tool: needle_expert
{'__arg1': 'date of the incident'}
🛠️ Called Tool: needle_expert
{'__arg1': 'location of the incident'}
   → Result (needle_expert):
124 Maple Street, Austin, TX 78701
   → Result (needle_expert):
November 16, 2024
🛠️ Called Tool: get_historical_weather
{'city': 'Austin', 'date': '2024-11-16'}
   → Result (get_historical_weather):
Weather Report for Austin on 2024-11-16:
- Max Temp: 27.0 °C
- Min Temp: 12.1 °C
- Precipitation: 0.0 mm
- Rain: 0.0 mm
- Snow: 0.0 cm
- Max Wind Speed: 18.4 km/h

The incident occurred at 124 Maple Street, Austin, TX 78701, on November 16, 2024.
On that day, the weather in Austin was dry with no precipitation, rain, or snow recorded.
The weather conditions do not appear to have contributed to the loss...
```

**Analysis**:

1.  **Chain of Thought**: The agent realized it couldn't check "weather for that date" without knowing _what_ date and location.
2.  **Multi-Step Execution**: It spawned two sub-tasks to `needle_expert` to get the context.
3.  **MCP Execution**: With the context (Austin, 2024-11-16), it correctly called the `get_historical_weather` MCP tool.
