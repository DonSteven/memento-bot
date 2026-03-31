"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    
    channel 把消息放进来
    agent 从里面取消息处理
    agent 处理完再把回复放出去
    channel 再把回复发送到外部平台
    外部用户 → channel → inbound queue → agent → outbound queue → channel → 外部用户
    """

    def __init__(self): # 没有消息时，agent 不应该忙等，所以用异步队列。异步不等于多线程，可以在同一线程上        
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue() # 消息来源，用户id，来源的id，消息内容，时间戳
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue() # 消息来源，来源的id，消息内容，回复对象

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg) # 告诉程序“这里可能要等，先把控制权让出去”

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    @property # 这个方法是属性，调用时不用写bus.inbound_size()，直接bus.inbound_size
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()
