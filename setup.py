from setuptools import find_packages, setup

setup(
    name="fraud_detection_service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "joblib",
        "xgboost",
        "numpy",
        "pandas"
    ],
    author="L.VijayVarma",
    description="End-to-End MLOps Fraud Detection Pipeline on macOS",
)
