from fastapi import FastAPI

app = FastAPI(
    title="Car Plate Analysis API",
    description="API do analizy tablic rejestracyjnych (Zaawansowane Programowanie / Dobre Praktyki)",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """
    Podstawowy endpoint powitalny.
    """
    return {
        "project": "Car Plate Analysis",
        "status": "Running",
        "message": "Witaj w systemie analizy obrazu!"
    }

@app.get("/health")
def health_check():
    """
    Sprawdza, czy serwer działa poprawnie.
    """
    return {"status": "healthy"}