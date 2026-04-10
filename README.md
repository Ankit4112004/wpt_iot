# EV WPT Monitor — Dashboard with ML Predictions

A real-time Wireless Power Transfer monitoring dashboard pulling live data from ThingSpeak with integrated machine learning predictions for battery efficiency.

## Features

✅ **Live Dashboard**
- Real-time metrics: Voltage, Current, Power, State of Charge
- Interactive charts (Voltage, Current, Power, SOC over time)
- Data table with raw feed logs
- Corner popup notifications for data updates
- Status indicators and connectivity monitoring

✅ **ML Efficiency Prediction** (NEW)
- Trained RandomForest model on Li-ion battery data
- Predicts efficiency based on: SOC, Voltage, Current, Battery Temp, Ambient Temp
- Real-time predictions display on dashboard

✅ **Responsive Design**
- Dark theme (cyberpunk/tech aesthetic)
- Mobile-friendly layout
- Smooth animations and transitions

---

## Quick Start

### 1. Setup Python Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Train ML model (one-time)
python train_model.py

# Start API server
python api.py
```

The API will run on `http://localhost:5000`

### 2. Open Dashboard

Go to **`index.html`** in your browser or deploy to Vercel:
```bash
git add .
git commit -m "Add ML predictions"
git push
```

Visit: **`https://wpt-iot.vercel.app`** (or your Vercel URL)

---

## Configuration

### ThingSpeak Setup

1. Create/configure your ThingSpeak channel with fields:
   - Field 1: Voltage (V)
   - Field 2: Current (A)
   - Field 3: Power (W)
   - Field 4: Battery Temperature (°C)
   - Field 5: Ambient Temperature (°C)
   - Field 6: State of Charge (%)

2. In the dashboard:
   - Enter your **Channel ID**
   - Enter your **Read API Key**
   - Map fields to corresponding metrics
   - Click **"Fetch Data"**

### Field Mapping Example

```
Voltage → Field 1
Current → Field 2
Power → Field 3
SOC → Field 6
Battery Temp → Field 4 (used for predictions)
Ambient Temp → Field 5 (used for predictions)
```

---

## Files

- **`index.html`** — Main dashboard (deployed to Vercel)
- **`api.py`** — Flask backend for ML predictions (run locally)
- **`train_model.py`** — ML model trainer (creates model.pkl & scaler.pkl)
- **`battery_efficiency_model.py`** — Detailed ML analysis script
- **`evdata2.csv`** — Training dataset (Li-ion battery charging data)
- **`requirements.txt`** — Python dependencies

---

## How It Works

### Dashboard Flow
1. User enters ThingSpeak credentials
2. Clicks **"Fetch Data"** or enables **"Auto (30s)"**
3. Dashboard fetches latest data from ThingSpeak API
4. Metrics cards update with live values
5. Charts display historical trends
6. **Bottom-right popup** shows data update notification
7. **ML Prediction Panel** displays predicted efficiency

### ML Prediction Flow
1. Dashboard extracts: SOC, Voltage, Current, Battery Temp, Ambient Temp
2. Sends data to local API (`http://localhost:5000/api/predict`)
3. API scales data using trained scaler
4. RandomForest model predicts efficiency
5. Prediction displayed in panel below metrics

---

## Model Details

- **Algorithm**: Random Forest Regressor (100 trees)
- **Training Data**: Li-ion batteries from evdata2.csv
- **Features**:
  - abs(SOC) — Absolute State of Charge (%)
  - Voltage (V)
  - Current (A)
  - Battery Temperature (°C)
  - Ambient Temperature (°C)
- **Target**: Efficiency (%)
- **Performance**: ~98% R² score on test data

---

## Deployment

### Option 1: Vercel (Recommended for Dashboard)

```bash
# Rename file for Vercel routing
mv minor_project.html index.html

# Push to GitHub
git add .
git commit -m "Deploy to Vercel with ML predictions"
git push

# Go to Vercel dashboard → Import repository → Deploy
```

⚠️ **Note**: The ML API (`api.py`) must run locally to provide predictions. 
The dashboard will still work without it, but predictions won't display.

### Option 2: Full Stack Deployment

For production with API predictions, use:
- **Frontend**: Vercel (index.html)
- **Backend**: Heroku, Railway, or DigitalOcean (api.py)

---

## Usage Example

### Manual Data Fetch
1. Enter Channel ID: `2345678`
2. Enter API Key: `XXXXXXXXXXXX`
3. Select Fields:
   - Voltage → Field 1
   - Current → Field 2
   - Power → Field 3
   - SOC → Field 6
4. Click **"Fetch Data"**
5. View metrics, charts, and predictions

### Auto-Refresh
1. Click **"Auto (30s)"** button
2. Dashboard fetches and predicts every 30 seconds
3. Notifications popup for each update
4. Click **"Stop Auto"** to disable

---

## Troubleshooting

### API Connection Error
```
⚠ API unavailable (run: python api.py)
```
**Solution**: Start the Flask API server in a terminal:
```bash
python api.py
```

### ThingSpeak.html Connection Error
```
Error: Invalid channel or API key
```
**Solution**: 
- Verify Channel ID is correct
- Verify API key has read permissions
- Check channel has data

### Dashboard Not Updating
- Ensure auto-refresh is enabled
- Check browser console for errors
- Verify ThingSpeak connectivity

---

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (vanilla)
- **Backend**: Python Flask, scikit-learn
- **Database**: ThingSpeak IoT cloud
- **Hosting**: Vercel (frontend), Local (backend)
- **ML Library**: scikit-learn (RandomForest, StandardScaler)

---

## License

Open source project for EV WPT monitoring and research.

---

## Support

For issues or questions:
1. Check the configuration section above
2. Review browser console errors (F12 → Console)
3. Ensure Python dependencies are installed
4. Verify ThingSpeak channel connectivity

---

**Last Updated**: April 11, 2026
**Version**: 1.1 (with ML predictions)
