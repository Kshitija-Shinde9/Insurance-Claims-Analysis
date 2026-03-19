# Plotly Dash Dashboard

## How to Run
```bash
cd dashboard
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:8050
```

---

## Dashboard Screenshots

### Overview - Header, Filters and KPI Cards
![WhatsApp Image 2026-03-18 at 9 27 23 AM](https://github.com/user-attachments/assets/2261ba4b-b6fe-4236-b19f-7426ee747e22)


### Charts - Segment, Policy, Premium and Risk Analysis
![WhatsApp Image 2026-03-18 at 9 27 23 AM (1)](https://github.com/user-attachments/assets/1538a9ef-d08d-486d-9505-05cb63ba2052)


---

## Features

| Panel | Description |
|---|---|
| Segmentation Group Filter | Filter by Segment1 to Segment5 |
| Policy Type Filter | Filter by Business, Family, Group, Individual |
| Risk Profile Filter | Filter by risk levels 0 to 3 |
| Year Range Slider | Filter policy start years from 2018 to 2023 |
| Filtered Customers | Total customers matching filters (max 53,503) |
| Average Premium | Mean premium amount across filtered records |
| Average Credit Score | Mean credit score (overall avg 673.26) |
| Average Policy Tenure | Mean policy duration in days (overall avg 1,097) |
| Customer Count by Segment | Bar chart - Segment5 largest (13,976), Segment1 smallest (8,683) |
| Policy Type Share | Donut chart - Group 34.1%, Business 26.1%, Family 23.2%, Individual 16.5% |
| Premium Amount by Segment | Box plot showing premium spread per segment |
| Credit Score Distribution | Overlaid histogram by segment (range 500 to 850) |
| Average Premium Heatmap | Risk profile x Segment matrix (range ~$2,917 to $3,110) |
| Policies Over Time | Line chart - peaked 2018 (~9,300), dipped 2019-2020, recovered 2021 |
