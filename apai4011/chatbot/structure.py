class DialogueStateTracker:
    """Tracks user intents and entities across the dialogue."""
    def __init__(self):
        self.intent = None  # Current user intent
        self.entities = {}  # Past entities (e.g., symptoms)

    def update_state(self, intent, entities):
        """Update the tracker with the current intent and entities."""
        self.intent = intent
        # Save entities only for diagnosis intents
        if intent == "ask_diagnosis":
            self.entities.update(entities)

    def get_past_entities(self):
        """Retrieve past entities to inform decision-making models."""
        return self.entities


class LanguageUnderstanding:
    """Handles input processing to extract intent and entities."""
    @staticmethod
    def process_input(user_input):
        # Mock logic for extracting intent and entities
        if "fever" in user_input.lower() or "cough" in user_input.lower():
            return "ask_diagnosis", {"symptoms": ["fever", "cough"]}
        elif "what is" in user_input.lower():
            if "flu" in user_input.lower():
                return "ask_info", {"disease": "flu"}
            elif "diabetes" in user_input.lower():
                return "ask_info", {"disease": "diabetes"}
        else:
            return "unknown", {}


class DiagnosisModel:
    """Determines disease based on symptoms."""
    @staticmethod
    def diagnose(symptoms):
        # Mock logic for diagnosis
        if "fever" in symptoms and "cough" in symptoms:
            return "flu"
        return None


class KnowledgeBase:
    """Fetches information about diseases, symptoms, and treatments."""
    @staticmethod
    def fetch_info(entity, query_type):
        # Mock logic for retrieving info
        knowledge = {
            "flu": {
                "description": "A viral infection that attacks the respiratory system.",
                "symptoms": "Fever, cough, sore throat, body aches.",
                "medication": "Antiviral drugs, rest, hydration.",
            },
            "diabetes": {
                "description": "A chronic condition that affects the way the body processes blood sugar.",
                "symptoms": "Frequent urination, increased thirst, unexplained weight loss.",
                "medication": "Insulin, metformin, lifestyle changes.",
            }
        }
        return knowledge.get(entity, {}).get(query_type, "Information not available.")


class ChatBot:
    """Combines all components to generate responses."""
    def __init__(self):
        self.tracker = DialogueStateTracker()
        self.lu = LanguageUnderstanding()
        self.diagnosis_model = DiagnosisModel()
        self.knowledge_base = KnowledgeBase()

    def handle_input(self, user_input):
        # Step 1: Process user input to extract intent and entities
        intent, entities = self.lu.process_input(user_input)

        # Step 2: Update the Dialogue State Tracker
        self.tracker.update_state(intent, entities)

        # Step 3: Handle different intents
        if intent == "ask_info":
            # Do not save the entity, only use current entity to fetch info
            entity = entities.get("disease")
            if entity:
                response = self.knowledge_base.fetch_info(entity, "description")
            else:
                response = "Please specify the disease you'd like information about."
        elif intent == "ask_diagnosis":
            # Save the entity and use both past and current entities for diagnosis
            symptoms = self.tracker.get_past_entities().get("symptoms", [])
            if symptoms:
                disease = self.diagnosis_model.diagnose(symptoms)
                if disease:
                    response = f"The symptoms match {disease}."
                else:
                    response = "Unable to determine the disease. Please provide more details."
            else:
                response = "Please specify your symptoms for a diagnosis."
        else:
            response = "I'm sorry, I didn't understand that."

        return response


# Example usage
if __name__ == "__main__":
    chatbot = ChatBot()

    # Simulated conversation
    print(chatbot.handle_input("What is diabetes?"))  # Expecting: "A chronic condition that affects the way the body processes blood sugar."
    print(chatbot.handle_input("I have fever and cough."))  # Expecting: "The symptoms match flu."
    print(chatbot.handle_input("What is flu?"))  # Expecting: "A viral infection that attacks the respiratory system."