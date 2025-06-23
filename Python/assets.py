from datetime import datetime

from feature_store.assets import Assets

if __name__ == "__main__":
    start_time = datetime.now()
    print("#" * 100)
    print("INICIO ASSETS")
    asset_generator = Assets()
    asset_generator.calendar_eop()
    asset_generator.calendar_date()
    asset_generator.generate_enrollment_base()
    end_time = datetime.now()
    print("FIN ASSETS")
    print(f"TERMINADO EN {end_time-start_time}")
