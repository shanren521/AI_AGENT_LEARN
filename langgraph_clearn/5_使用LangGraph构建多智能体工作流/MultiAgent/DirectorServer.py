from Director import graph
import uuid

config = {"configurable": {"thread_id": uuid.uuid4()}}

# 请讲一个笑话
# 请给一个上海到杭州的路线规划
# 金榜题名日
query = input("请输入: ")

res = graph.invoke({"messages": [query]}, config, stream_mode="values")
print(res["messages"][-1].content)