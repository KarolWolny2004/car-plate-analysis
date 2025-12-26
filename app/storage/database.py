from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Ścieżka do pliku bazy danych
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# Engine - silnik, który rozmawia z bazą
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Sesja - nasze "okienko" do wykonywania zapytań
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Klasa bazowa, po której będą dziedziczyć nasze modele
Base = declarative_base()

# Funkcja pomocnicza (Dependency), która otwiera i zamyka sesję
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()