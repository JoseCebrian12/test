from datetime import datetime

from feature_store.cobranzas.cobranzas import CobranzasClass

cbr = CobranzasClass()
cbr.REFACTOR_PERIOD = 202402


def bronze_to_silver():
    # cbr.check_refactor()

    print("\nBRONZE_TO_SILVER: RAW")
    cbr.bronze_to_silver_raw()

    print("\nSILVER_TO_SILVER: DOCUMENT LEVEL EOP")
    cbr.silver_to_silver_doc_eop()

    print("\nSILVER_TO_SILVER: DOCUMENT LEVEL DATE")
    cbr.silver_to_silver_doc_date()


def silver_to_gold():
    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP")
    cbr.silver_to_gold_eop()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL EOP - PERFIL MORISIDAD")
    cbr.silver_to_gold_eop_pm()

    print("\nSILVER_TO_GOLD: WINDOW AGGREGATION FEATURE LEVEL EOP")
    cbr.gold_eop_to_window_aggregation()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL DATE")
    cbr.silver_to_gold_date()

    print("\nSILVER_TO_GOLD: FEATURE LEVEL DATE - PERFIL MORISIDAD")
    cbr.silver_to_gold_date_pm()


if __name__ == "__main__":
    print("#" * 100)
    print(f"INICIO COBRANZAS")
    start_time = datetime.now()
    bronze_to_silver()
    silver_to_gold()
    end_time = datetime.now()
    print(f"\nRUTINA COBRANZAS TERMINADO EN {end_time-start_time}")