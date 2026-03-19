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

![plotly_1](https://github.com/user-attachments/assets/cccd57dd-f4f9-4a80-8472-0abcce34e221)

### Overview - Header, Filters and KPI Cards with filter change

<img width="1280" height="832" alt="image" src="https://github.com/user-attachments/assets/4c605612-f868-4a4c-b20d-5a8ecdb8582d" />



### Charts - Segment, Policy, Premium and Risk Analysis

![plotly2](https://github.com/user-attachments/assets/36105e7c-c652-423e-9a0b-967f8d061aff)

### Charts - Segment, Policy, Premium and Risk Analysis

<img width="1280" height="832" alt="image" src="https://github.com/user-attachments/assets/07450576-a5d3-4f4a-941b-1fa986bdf0e3" />

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
