I am building a restaurant order management AI agent system using LangGraph.

I will explain very clearly what I want and what the architecture is. I will also explain the nodes, the state, and the edges. In the end, I will give you test cases to simulate and tell me whether they pass or not.

The rough plan is:

There will be a user. When the code runs, the user will give an order. This will be taken as an input that I will type.

This order should go to the LLM. The order, as of now, should contain the dish and the required quantity.

For simplicity, for now we will limit the order to one particular dish and whatever quantity the user wants.

The LLM will have to extract the order name and quantity from the user input.

If the user input is unrelated to food ordering, the LLM should not process it and should tell the user that it is an AI agent for food ordering and not a general-purpose LLM.

Once the LLM has the order dish and quantity, it will send that to a node called `order_confirm`.

The task of `order_confirm` is to look at the menu (I will tell you the menu later in this prompt).

Then this node will decide one of three cases:

1. The order is available.
2. The order is partially available, meaning the quantity is not sufficient.
3. The order is not available at all (the dish is not in the menu or the available quantity is 0).

It will put this in the `status` of the state. (I will tell you the exact content of the LangGraph state as well.)

Once the LLM receives this:

* If the status is `confirmed` (fully available), it should call another node called `cook`.
* If the status is `partial` or `not available`, it should again prompt the user to decide.

The user can either place a new order or confirm if they want to go ahead with the partial order.

These order retries will be limited to 3 attempts, meaning if after 3 attempts the user is not satisfied, the system will come to the `END` node.

Now, when the `cook` node is called, there can be 2 cases:

* Either the cook is done, then the status will be `READY`.
* Or the cook fails (we can use a probability function). Give it a 40% chance of failure and a 60% chance of success.

If the cook fails, there should be 1 more attempt allowed for the cook to succeed. If the cook fails even after this retry, then the LLM should issue an apology to the user and come to the `END` state.

If the cook succeeds, the status should be `READY`, and the next node will be called, which is `serve`.

Similar to `cook`, this also has 2 cases:

* Serve passes.
* Serve fails.

This also has 2 retry attempts.

If the serve fails 2 times, then the LLM should issue an apology to the user and come to the `END` state.

If the serve succeeds, the status should become `COMPLETE`. The LLM should issue a message to the user saying that their order is complete.

If serve fails, then `cook` should be called one more time to retry.

Note that if cook has exhausted its retry attempts, then it should not cook again, and the LLM should issue an apology and come to the `END` state.

Now, the state of LangGraph:

There should be an annotated message between the LLM and the user.

There should be order details:

* Dish name as `str`
* Required quantity as `int`
* Available quantity as `str`

`order_confirm` will write the available quantity by reading the menu.

The LLM should get to know the order confirmation status by reading the state.

If a dish is not available in the menu, then `order_confirm` should write `0` as the available quantity.

Then there should be `status`.

Each node will update the status as specified in the above rules.

Then there should be:

* Order retry attempts: `3`
* Cook retry attempts: `2`
* Serve retry attempts: `2`

Each time a failure happens and a node is retrying, it should decrement the corresponding counter.

If any retry counter becomes `0`, it means that the retries are exhausted. The LLM should understand whether it has to give a retry or issue an apology by reading this counter.

In the end, there should be a `final_result` indicating whether the order was completed or not.

Write the code and ask if there are any open questions from your side. If there are, ask them.

I will give you some test scenarios to test later on.

### TC1

A user asks an unrelated question.

Then he places an order which is partially available.

Expected:

* The system should ask whether he wants to accept the partial order or place a new order.
* He rejects the partial order and wants to order again.
* Again, he orders a dish which is not available.
* Expected result: `END` due to order retry exhaustion.

### TC2

The user places an order which is fully available.

* Cook fails once.
* Cook retries and succeeds.
* Serve fails once.
* Cook retries and succeeds.
* Serve succeeds.
* Overall result: `SUCCESS`.

### TC3

The user orders a partially available item.

* He does not want the partial order.
* He orders again.
* The second order is fully available.
* Cook fails.
* Cook retries and succeeds.
* Serve fails.
* Cook retries and succeeds.
* Serve fails again.
* Cook does not retry because the cook retry attempts are exhausted.
* Overall result: `FAIL`.


I am using groq model, openai/gpt-oss-120b and key is in my env file
Make a small menu on your own of 5-7 items 