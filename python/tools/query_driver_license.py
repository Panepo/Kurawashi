def query_driver_license(number: str, state: str | None = None) -> dict[str, str | None]:
    return {"driver_license_number": number, "state": state}
