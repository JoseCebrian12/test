from datetime import datetime

from feature_store.inasistencias.inasistencias import InasistenciaClass

inasist = InasistenciaClass()
inasist.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    inasist.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    inasist.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: EOP PROCESS")
    inasist.silver_to_silver_eop()

    print("\nSILVER_TO_SILVER: DATE PROCESS")
    inasist.silver_to_silver_date()


def silver_to_gold():
    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    inasist.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL DATE")
    inasist.silver_to_gold_date()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO INASISTENCIAS")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA INASISTENCIAS TERMINADO EN {end_time-start_time}")
