from app.utils.contact_extraction import answer_contact_query


def test_answer_contact_query_extracts_email_and_urls():
    answer = answer_contact_query(
        "can you fetch email or url?",
        [
            {
                "text": "Syed Faisal Ali Shah +92-3445490208 faisalalishah007@gmail.com linkedin.com/faisal-ali-shah github.com/faisalali"
            }
        ],
    )

    assert "Email: faisalalishah007@gmail.com" in answer
    assert "linkedin.com/faisal-ali-shah" in answer
    assert "github.com/faisalali" in answer
