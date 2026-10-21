import json

from foreman.core.reconfiguration_protocol import (
    ASSIGN,
    ReconfigurationMessage,
)


def test_reconfiguration_envelope_round_trips():
    message = ReconfigurationMessage(ASSIGN, {"shard": "s0"}, epoch=4)
    decoded = ReconfigurationMessage.from_dict(json.loads(json.dumps(message.to_dict())))
    assert decoded == message


def test_legacy_messages_are_not_claimed():
    assert ReconfigurationMessage.from_dict(
        {"type": "assign_task", "data": {}, "job_id": "job"}
    ) is None
