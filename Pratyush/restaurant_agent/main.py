import os
import random
import re
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

# Load environment variables such as GROQ_API_KEY from a .env file.
load_dotenv()

# Small restaurant menu used by the agent.
MENU = {
    "margherita pizza": 10,
    "butter chicken": 8,
    "veg biryani": 6,
    "pasta alfredo": 5,
    "paneer tikka": 7,
    "cold coffee": 12,
}


# TypedDict is the state schema for LangGraph.
# It tells the graph exactly which fields exist in the state.
class OrderDetails(TypedDict):
    dish: str
    quantity: int
    available_quantity: int


class RestaurantState(TypedDict):
    messages: list[BaseMessage]
    order: OrderDetails
    status: str
    order_retry_attempts: int
    cook_retry_attempts: int
    serve_retry_attempts: int
    final_result: str


# This is how we create the language model connection.
# A Groq API key is required to call the LLM in the real environment.
# If it is missing, we fall back to a local parser so the project still runs.
def get_llm() -> ChatGroq | None:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, api_key=groq_api_key)


# A simple normalizer so we can compare text reliably.
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


# This helper matches the user text to an item from the menu.
def match_menu_dish(text: str) -> str | None:
    normalized = normalize_text(text)
    for dish in MENU:
        if dish in normalized:
            return dish
    for dish in MENU:
        tokens = set(dish.split())
        if tokens.issubset(set(normalized.split())):
            return dish
    return None


# The LLM extraction is helpful when a real Groq API key exists.
# If not, we still parse text using rules, which keeps the project working.
def extract_order_from_text(user_input: str) -> dict[str, Any]:
    llm = get_llm()
    if llm is not None:
        try:
            prompt = (
                "You are a restaurant ordering parser. Extract order details from the user's message. "
                "If the message is unrelated to food ordering, return {'valid': false, 'reason': 'not a food order'}. "
                "Otherwise return JSON with 'valid': true, 'dish': dish name, 'quantity': quantity as integer. "
                "Dish must exactly match one of these menu items: margherita pizza, butter chicken, veg biryani, pasta alfredo, paneer tikka, cold coffee."
            )
            response = llm.invoke(prompt + "\nUser input: " + user_input)
            content = str(response.content)
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                parsed = match.group(0)
                clean = parsed.replace("```", "").strip()
                payload = eval(clean, {"__builtins__": {}}, {})
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass

    # Fallback rule-based parser.
    normalized = normalize_text(user_input)
    food_indicators = [
        "order",
        "want",
        "need",
        "buy",
        "please",
        "pizza",
        "chicken",
        "biryani",
        "pasta",
        "paneer",
        "coffee",
        "dish",
    ]
    if not any(word in normalized for word in food_indicators):
        return {"valid": False, "reason": "not a food order"}

    matched_dish = match_menu_dish(normalized)
    if matched_dish is None:
        return {"valid": False, "reason": "dish not found"}

    quantity_match = re.search(r"(\d+)", normalized)
    quantity = int(quantity_match.group(1)) if quantity_match else 1

    return {"valid": True, "dish": matched_dish, "quantity": quantity}


# A fresh state is created for each new order run.
def make_initial_state(user_input: str) -> RestaurantState:
    return {
        "messages": [HumanMessage(content=user_input)],
        "order": {"dish": "", "quantity": 0, "available_quantity": 0},
        "status": "new",
        "order_retry_attempts": 3,
        "cook_retry_attempts": 2,
        "serve_retry_attempts": 2,
        "final_result": "PENDING",
    }


# A node is a function that receives the current state and returns a new state.
# In LangGraph, nodes are just small, single-purpose functions.
def parse_order_node(state: RestaurantState) -> RestaurantState:
    user_text = state["messages"][-1].content if state["messages"] else ""
    parsed = extract_order_from_text(user_text)

    if not parsed.get("valid"):
        state["status"] = "invalid"
        state["final_result"] = "FAIL"
        state["messages"].append(
            AIMessage(
                content="I am a food-ordering AI agent. Please ask me about dishes and quantities from the restaurant menu."
            )
        )
        return state

    dish = str(parsed["dish"]).strip()
    quantity = int(parsed["quantity"])
    state["order"] = {"dish": dish, "quantity": quantity, "available_quantity": 0}
    state["status"] = "order_extracted"
    state["messages"].append(AIMessage(content=f"Order captured: {dish} x {quantity}."))
    return state


# order_confirm decides if the requested quantity is available, partial, or unavailable.
def order_confirm_node(state: RestaurantState) -> RestaurantState:
    dish = state["order"]["dish"].lower()
    requested_qty = state["order"]["quantity"]
    available_qty = MENU.get(dish, 0)

    # If dish is missing from the menu, we set available quantity to 0.
    if available_qty == 0:
        state["order"]["available_quantity"] = 0
        state["status"] = "not_available"
        state["messages"].append(
            AIMessage(content=f"{dish.title()} is not available in the menu. Available quantity: 0.")
        )
        return state

    if requested_qty <= available_qty:
        state["order"]["available_quantity"] = available_qty
        state["status"] = "confirmed"
        state["messages"].append(
            AIMessage(content=f"{dish.title()} is available. Stock: {available_qty}. Order can be confirmed.")
        )
        return state

    # Partial order means stock exists but not enough for the full request.
    state["order"]["available_quantity"] = available_qty
    state["status"] = "partial"
    state["messages"].append(
        AIMessage(
            content=(
                f"Only {available_qty} units of {dish.title()} are currently available. "
                "Would you like to accept the partial order or place a new order?"
            )
        )
    )
    return state


# This node handles the user decision after partial or unavailable order.
# It reduces the order retry counter each time the user chooses to retry.
def order_issue_node(state: RestaurantState, user_decision: str | None = None) -> RestaurantState:
    status = state["status"]
    if status not in {"partial", "not_available"}:
        return state

    if user_decision == "accept_partial" and status == "partial":
        state["order"]["quantity"] = state["order"]["available_quantity"]
        state["status"] = "confirmed"
        state["messages"].append(AIMessage(content="The partial order has been accepted and will proceed."))
        return state

    # If order retry attempts reach 0, the process must end.
    if state["order_retry_attempts"] <= 0:
        state["status"] = "END"
        state["final_result"] = "FAIL"
        state["messages"].append(AIMessage(content="Order retry limit exhausted. Ending the order process."))
        return state

    state["order_retry_attempts"] -= 1

    if user_decision in {"new_order", None}:
        state["status"] = "awaiting_new_order"
        state["messages"].append(
            AIMessage(
                content=(
                    f"Please place a new order. Remaining order retry attempts: {state['order_retry_attempts']}."
                )
            )
        )
        return state

    state["status"] = "END"
    state["final_result"] = "FAIL"
    return state


# cook_node simulates the kitchen activity.
# A normal probability function is used: 60% success, 40% failure.
def cook_node(state: RestaurantState, outcome: bool | None = None) -> RestaurantState:
    if outcome is None:
        outcome = random.random() < 0.6

    if outcome:
        state["status"] = "READY"
        state["messages"].append(AIMessage(content="The cook succeeded. Your dish is ready."))
        return state

    # If cooking fails, we spend one cook retry.
    state["cook_retry_attempts"] -= 1
    state["status"] = "cook_failed"

    if state["cook_retry_attempts"] <= 0:
        state["status"] = "END"
        state["final_result"] = "FAIL"
        state["messages"].append(AIMessage(content="Sorry, the cook could not complete the order after all attempts."))
        return state

    state["messages"].append(
        AIMessage(
            content=(
                f"The cook failed. Retrying cook. Remaining cook attempts: {state['cook_retry_attempts']}."
            )
        )
    )
    return state


# serve_node simulates serving the food to the customer.
# It has its own retry counter and can trigger a cook retry if service fails.
def serve_node(state: RestaurantState, outcome: bool | None = None) -> RestaurantState:
    if outcome is None:
        outcome = random.random() < 0.6

    if outcome:
        state["status"] = "COMPLETE"
        state["final_result"] = "SUCCESS"
        state["messages"].append(AIMessage(content="Your order is complete and has been served successfully."))
        return state

    state["serve_retry_attempts"] -= 1
    state["status"] = "serve_failed"

    if state["serve_retry_attempts"] <= 0:
        state["status"] = "END"
        state["final_result"] = "FAIL"
        state["messages"].append(AIMessage(content="Sorry, serving failed after all attempts."))
        return state

    state["messages"].append(
        AIMessage(
            content=(
                f"Serving failed. Retrying service. Remaining serve attempts: {state['serve_retry_attempts']}."
            )
        )
    )
    return state


# Conditional edges are a key LangGraph concept.
# They let the graph choose the next node based on the current state.
def route_after_parse(state: RestaurantState) -> str:
    if state["status"] == "order_extracted":
        return "order_confirm"
    return "END"


# A state may contain a partial or unavailable order.
# In that case the graph asks the human to either accept or retry.
def route_after_confirm(state: RestaurantState) -> str:
    if state["status"] == "confirmed":
        return "cook"
    if state["status"] in {"partial", "not_available"}:
        return "order_issue"
    return "END"


# This route decides whether the process is still active or should end.
def route_after_issue(state: RestaurantState) -> str:
    if state["status"] == "awaiting_new_order":
        return "parse_order"
    if state["status"] == "confirmed":
        return "cook"
    return "END"


# After a cook attempt, we either continue to serving, retry cooking, or end.
def route_after_cook(state: RestaurantState) -> str:
    if state["status"] == "READY":
        return "serve"
    if state["status"] == "cook_failed":
        return "cook" if state["cook_retry_attempts"] > 0 else "END"
    return "END"


# After a serve attempt, we either finish, retry cook, or end.
def route_after_serve(state: RestaurantState) -> str:
    if state["status"] == "COMPLETE":
        return "END"
    if state["status"] == "serve_failed":
        return "cook" if state["serve_retry_attempts"] > 0 else "END"
    return "END"


# The builder pattern is the simplest way to define a graph in LangGraph.
# It is very similar to the example you shared in implementationLangGraph.
def build_restaurant_graph() -> StateGraph:
    builder = StateGraph(RestaurantState)

    # Add nodes.
    builder.add_node("parse_order", parse_order_node)
    builder.add_node("order_confirm", order_confirm_node)
    builder.add_node("order_issue", order_issue_node)
    builder.add_node("cook", cook_node)
    builder.add_node("serve", serve_node)

    # Start point of the graph.
    builder.set_entry_point("parse_order")

    # The graph can branch based on the state value.
    builder.add_conditional_edges("parse_order", route_after_parse, {"order_confirm": "order_confirm", "END": END})
    builder.add_conditional_edges("order_confirm", route_after_confirm, {"cook": "cook", "order_issue": "order_issue", "END": END})
    builder.add_conditional_edges("order_issue", route_after_issue, {"parse_order": "parse_order", "cook": "cook", "END": END})
    builder.add_conditional_edges("cook", route_after_cook, {"serve": "serve", "cook": "cook", "END": END})
    builder.add_conditional_edges("serve", route_after_serve, {"END": END, "cook": "cook"})

    builder.add_edge("serve", END)
    return builder.compile()


class RestaurantOrderAgent:
    def __init__(self) -> None:
        self.graph = build_restaurant_graph()

    def run(self, user_input: str, decision: str | None = None) -> RestaurantState:
        state = make_initial_state(user_input)
        state = self.graph.invoke(state)

        # The graph is enough for the main execution flow.
        # For partial orders, we allow the caller to provide a decision.
        if state["status"] in {"partial", "not_available"} and decision:
            state = order_issue_node(state, decision)

        return state


# These test scenarios reflect the cases you described in the prompt.
def run_test_cases() -> None:
    agent = RestaurantOrderAgent()

    # TC1: unrelated question, then partial order rejected, then unavailable order -> END due to retry exhaustion.
    state1 = agent.run("What is the weather today?")
    state1 = agent.run("I want 2 veg biryani")
    state1 = order_issue_node(state1, "new_order")
    state1 = make_initial_state("I want 5 butter chicken")
    state1 = parse_order_node(state1)
    state1 = order_confirm_node(state1)
    state1 = order_issue_node(state1, "new_order")
    state1 = order_issue_node(state1, "new_order")
    print("TC1 result:", state1["final_result"], "Status:", state1["status"])

    # TC2: fully available order, one cook fail then success, one serve fail then cook success then serve success.
    state2 = make_initial_state("I want 2 margherita pizza")
    state2 = parse_order_node(state2)
    state2 = order_confirm_node(state2)
    state2 = cook_node(state2, outcome=False)
    state2 = cook_node(state2, outcome=True)
    state2 = serve_node(state2, outcome=False)
    state2 = cook_node(state2, outcome=True)
    state2 = serve_node(state2, outcome=True)
    print("TC2 result:", state2["final_result"], "Status:", state2["status"])

    # TC3: partial, then new order full, cook fails, retry succeeds, serve fails, retry cook succeeds, serve fails again, no more cook retry.
    state3 = make_initial_state("I want 3 paneer tikka")
    state3 = parse_order_node(state3)
    state3 = order_confirm_node(state3)
    state3 = order_issue_node(state3, "new_order")

    state3 = make_initial_state("I want 2 margherita pizza")
    state3 = parse_order_node(state3)
    state3 = order_confirm_node(state3)
    state3 = cook_node(state3, outcome=False)
    state3 = cook_node(state3, outcome=True)
    state3 = serve_node(state3, outcome=False)
    state3 = cook_node(state3, outcome=True)
    state3 = serve_node(state3, outcome=False)
    state3 = cook_node(state3, outcome=False)
    print("TC3 result:", state3["final_result"], "Status:", state3["status"])


if __name__ == "__main__":
    run_test_cases()
