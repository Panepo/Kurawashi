def query_license_plate(number: str, state: str | None) -> dict[str, str | None]:
    return {"license_plate_number": number, "state": state}
