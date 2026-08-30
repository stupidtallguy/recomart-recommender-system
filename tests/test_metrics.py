from src.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
)


def test_recall():

    recs = [
        1,2,3,4,5
    ]

    truth = {
        1,5,10
    }


    assert recall_at_k(
        recs,
        truth,
        5
    ) == 2/3



def test_precision():

    recs = [
        1,2,3,4,5
    ]

    truth = {
        1,5,10
    }


    assert precision_at_k(
        recs,
        truth,
        5
    ) == 2/5



def test_ndcg():

    recs = [
        1,2,3
    ]

    truth = {
        1,3
    }


    score = ndcg_at_k(
        recs,
        truth,
        3
    )


    assert score > 0