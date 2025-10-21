from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import joblib, os, datetime as dt

app = FastAPI()

class PredictRequest(BaseModel):
    test_csv_path: str

def mm(x: pd.Series) -> pd.Series:
    return (x - x.min()) / (x.max() - x.min())

@app.post("/predict")
def predict(req: PredictRequest):
    test_path = Path(req.test_csv_path)
    if not test_path.exists():
        raise HTTPException(status_code=404)

    model_path = Path("artifacts/best_pipeline.pkl")
    if not model_path.exists():
        raise HTTPException(status_code=500)

    try:
        fitted = joblib.load(model_path)
    except Exception as e:
        raise HTTPException(status_code=500)

    try:
        df = pd.read_csv(test_path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    except Exception as e:
        raise HTTPException(status_code=400)

    if "id" not in df.columns:
        raise HTTPException(status_code=400)
    ids = df["id"].copy()

    drop_cols = ["Unnamed: 0","id","Medication Use","Troponin","CK-MB","Income","Blood sugar","Smoking"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    str_cols = ["Diabetes","Family History","Smoking","Obesity","Alcohol Consumption","Diet","Previous Heart Problems","Gender"]
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype("str")

    if "Heart Attack Risk (Binary)" in df.columns:
        df["Heart Attack Risk (Binary)"] = df["Heart Attack Risk (Binary)"].astype(int)

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
        mask = ~df["Gender"].isin(["0","1",0,1,"0.0","1.0"])
        df = df.loc[mask].copy()
        ids = ids.loc[mask].copy()
        df["Gender"] = df["Gender"].replace({"male": 0, "m": 0, "female": 1, "f": 1})
        df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce")
        df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0]).astype(int)

    df["Mean Pressure"] = (df["Systolic blood pressure"] + 2*df["Diastolic blood pressure"]) / 3
    df["Heart Mp"] = df["Heart rate"] / (df["Mean Pressure"] + 1e-6)
    df["PP_to_SBP"] = (df["Systolic blood pressure"] - df["Diastolic blood pressure"]) / (df["Systolic blood pressure"] + 1e-6)
    df["CHOLxTG"] = df["Cholesterol"] * df["Triglycerides"]
    df["Exercise_per_Day"] = df["Exercise Hours Per Week"] / 7.0
    df["Sedentary_to_ActivityDays"] = df["Sedentary Hours Per Day"] / (df["Physical Activity Days Per Week"] + 1e-6)
    df["Stress_to_Sleep"] = df["Stress Level"] / (df["Sleep Hours Per Day"] + 1e-6)
    df["Age_BMI"] = df["Age"] * df["BMI"]

    to_num = ["Alcohol Consumption","Obesity","Diet","Previous Heart Problems","Exercise Hours Per Week","Physical Activity Days Per Week","Sedentary Hours Per Day","Stress Level","Sleep Hours Per Day","Family History","Diabetes"]
    for c in to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    df["Lifestyle Index"] = df["Alcohol Consumption"] - df["Obesity"] - df["Previous Heart Problems"] + (1 - mm(df["Diet"]))
    df["Activity Index"] = mm(df["Exercise Hours Per Week"]) + mm(df["Physical Activity Days Per Week"]) - mm(df["Sedentary Hours Per Day"])
    df["Stress Index"] = mm(df["Stress Level"]) - mm(df["Sleep Hours Per Day"])
    df["Str Act Index"] = df["Activity Index"] * df["Stress Index"]
    df["Health Index"] = df["Family History"] - df["Previous Heart Problems"] - df["Diabetes"] + df["Activity Index"]
    df = df.drop(columns=[c for c in to_num if c in df.columns])

    try:
        proba = fitted.predict_proba(df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500)

    y_hat = (proba >= 0.25).astype(int)
    out = pd.DataFrame({
        "id": ids.reset_index(drop=True),
        "prediction": pd.Series(y_hat).reset_index(drop=True)
    })

    os.makedirs("artifacts", exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("artifacts") / f"predictions_{ts}.csv"
    out.to_csv(out_path, index=False, sep=';')

    return FileResponse(str(out_path), media_type="text/csv", filename=out_path.name)

@app.get("/")
def root():
    return {"name": "heart-risk-api", "usage": "POST /predict with {'test_csv_path': './test.csv'}"}