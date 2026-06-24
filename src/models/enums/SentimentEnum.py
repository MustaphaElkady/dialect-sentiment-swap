from enum import Enum

class SentimentEnum(str,Enum):
    NIGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"