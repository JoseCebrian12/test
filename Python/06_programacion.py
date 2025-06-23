from datetime import datetime

from feature_store.programacion.programacion import ProgramacionClass

progr = ProgramacionClass()
progr.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    progr.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    progr.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP PROCESS")
    progr.silver_to_silver_eop()

    print("\nSILVER_TO_SILVER: DATE PROCESS")
    progr.silver_to_silver_date()


def silver_to_gold():
    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    progr.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL DATE")
    progr.silver_to_gold_date()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO PROGRAMACION")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA PROGRAMACION TERMINADO EN {end_time-start_time}")
