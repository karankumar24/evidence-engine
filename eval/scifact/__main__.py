"""Enables: python -m eval.scifact run --split dev ..."""
if __name__ == "__main__":
    from eval.scifact.cli import main  # noqa: PLC0415 — deferred import
    main()
