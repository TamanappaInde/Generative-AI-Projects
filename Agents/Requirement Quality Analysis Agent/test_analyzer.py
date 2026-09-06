from analyzer import LocalRetriever, analyze_requirements, parse_requirements


def test_parse_requirements_preserves_explicit_ids_and_numbered_ids():
    requirements = parse_requirements("BR-001: As an admin, the system must export data.\n2. Users can search.")

    assert [item.requirement_id for item in requirements] == ["BR-001", "REQ-2"]


def test_analysis_finds_near_duplicates_and_missing_acceptance_criteria():
    result = analyze_requirements(
        "BR-001: As a customer, the user must reset a password quickly.\n"
        "BR-002: As a customer, the user must reset their password quickly.",
        expected_roles=["Customer"],
    )

    categories = {finding.category for finding in result.findings}
    assert "Near duplicate" in categories
    assert "Missing acceptance criteria" in categories
    assert result.summary["roles_detected"] == ["Customer"]


def test_local_retriever_returns_relevant_requirement():
    result = analyze_requirements("BR-001: As an admin, the system must export reports.")

    retrieved = LocalRetriever(result.requirements).search("export reports")
    assert retrieved[0].requirement_id == "BR-001"