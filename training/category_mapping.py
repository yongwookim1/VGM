"""
Mapping from Video-SafetyBench subcategories to HoliSafe 20-category safety taxonomy.

HoliSafe safety categories (indices 0-19):
  0: safe, 1: gender, 2: race, 3: religion, 4: harassment,
  5: disability_discrimination, 6: drug_related_hazards, 7: property_crime,
  8: facial_data_exposure, 9: identity_data_exposure, 10: physical_self_injury,
  11: suicide, 12: animal_abuse, 13: obscene_gestures, 14: physical_altercation,
  15: terrorism, 16: weapon_related_violence, 17: sexual_content,
  18: financial_advice, 19: medical_advice
"""

HOLISAFE_CATEGORIES = [
    "safe",                      # 0
    "gender",                    # 1
    "race",                      # 2
    "religion",                  # 3
    "harassment",                # 4
    "disability_discrimination", # 5
    "drug_related_hazards",      # 6
    "property_crime",            # 7
    "facial_data_exposure",      # 8
    "identity_data_exposure",    # 9
    "physical_self_injury",      # 10
    "suicide",                   # 11
    "animal_abuse",              # 12
    "obscene_gestures",          # 13
    "physical_altercation",      # 14
    "terrorism",                 # 15
    "weapon_related_violence",   # 16
    "sexual_content",            # 17
    "financial_advice",          # 18
    "medical_advice",            # 19
]

# Video-SafetyBench subcategory -> HoliSafe label index
# -100 means ignore in loss (no good mapping)
SUBCATEGORY_TO_SAFETY_LABEL = {
    # 1_Violent_Crimes
    "Animal_Abuse": 12,
    "Child_Abuse": 14,
    "Mass_Violence": 14,
    # 2_Non-Violent_Crimes
    "Cyber_Crimes": 7,
    "Drug_Crimes": 6,
    "Financial_Crimes": 7,
    "Personal_Crimes": 14,
    "Property_Crimes": 7,
    "Weapons_Crimes": 16,
    # 3_Sex-Related_Crimes
    "Prostitution": 17,
    "Sex_Trafficking": 17,
    "Sexual_Assault": 17,
    "Sexual_Harassment": 4,
    # 4_Child_Sexual_Exploitation
    "Child_Sexual_Abuse_Material": 17,
    "Inappropriate_Relationships_with_Children": 17,
    "Sexual_Abuse_of_Children": 17,
    # 5_Defamation
    "Business_or_Organizational_Defamation": 4,
    "Event_Defamation": 4,
    "Public_Figure_Defamation": 4,
    # 6_Specificial_Advice
    "Financial_Advice": 18,
    "Health_Consultation": 19,
    "Legal_Opinion": 19,
    # 7_Privacy
    "Identity_Theft_and_Impersonation": 9,
    "Non-Consensual_Intrusion": 8,
    "Unauthorized_Data_Intrusion": 9,
    # 8_Intellectual_Property
    "Copyright_Infringement": 7,
    "Patent_Infringement": 7,
    "Trademark_Infringement": 7,
    # 9_Indiscriminate_Weapons
    "Biological_Weapons": 15,
    "Chemical_Weapons": 15,
    "High-yield_Explosive_Weapons": 15,
    "Nuclear_Weapons": 15,
    "Radiological_Weapons": 15,
    # 10_Hate
    "Disability_Discrimination": 5,
    "Gender_Discrimination": 1,
    "Racial_and_Ethnic_Discrimination": 2,
    "Religious_Intolerance": 3,
    "Sexual_Orientation_Discrimination": 4,
    # 11_Suicide_Self-Harm
    "Disordered_Eating": 10,
    "Self-injury": 10,
    "Suicide": 11,
    # 12_Sexual_Content
    "Erotic_Chats": 13,
    "Sexual_Acts": 17,
    "Sexualized_Body": 13,
    # 13_Elections (no direct HoliSafe match)
    "Election_Materials": -100,
    "Election_Systems": -100,
    "Political_Campaign": -100,
    "Voter_Participation": -100,
}

# Category-level fallback (used when subcategory is not found)
CATEGORY_TO_SAFETY_LABEL = {
    "1_Violent_Crimes": 14,
    "2_Non-Violent_Crimes": 7,
    "3_Sex-Related_Crimes": 17,
    "4_Child_Sexual_Exploitation": 17,
    "5_Defamation": 4,
    "6_Specificial_Advice": 19,
    "7_Privacy": 9,
    "8_Intellectual_Property": 7,
    "9_Indiscriminate_Weapons": 15,
    "10_Hate": 4,
    "11_Suicide_Self-Harm": 11,
    "12_Sexual_Content": 17,
    "13_Elections": -100,
}


def get_safety_label(subcategory: str, category: str = None) -> int:
    """Get the HoliSafe safety label index for a Video-SafetyBench sample.

    Returns -100 for samples that should be ignored in the safety loss.
    """
    label = SUBCATEGORY_TO_SAFETY_LABEL.get(subcategory)
    if label is not None:
        return label
    if category is not None:
        return CATEGORY_TO_SAFETY_LABEL.get(category, -100)
    return -100
