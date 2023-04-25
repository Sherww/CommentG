
# question_types = ['are', 'bes', 'can', 'dos', 'forpurpose', 'have', 'howmany', 'howmuch', 'hows', 'inwhich',
#                     'iss', 'may', 'must', 'should', 'whats', 'whens', 'wheres', 'whichs', 'whos', 'whys', 'wills', 'other']
# question_to_id = {q: i for i, q in enumerate(question_types)}

# def get_question_type(question: str):
#     token_list = question.strip().split(' ')
#     if len(token_list) < 3:
#         return 'other'
    
#     first_token = token_list[0].lower()
#     second_token = token_list[1].lower()
#     third_token = token_list[2].lower()

#     if first_token == 'are':
#         return 'are'
#     elif first_token in ['be', 'being', 'been']:
#         return 'bes'
#     elif first_token in ['can', 'could']:
#         return 'can'
#     elif first_token in ['do', 'did', 'does']:
#         return 'dos'
#     elif first_token == 'for' and second_token == 'what' and third_token == 'purpose':
#         return 'forpurpose'
#     elif first_token in ['have', 'has', 'had']:
#         return 'have'
#     elif 'how many' in question:
#         return 'howmany'
#     elif 'how much' in question:
#         return 'howmuch'
#     elif first_token == 'how':
#         return 'hows'
#     elif first_token == 'in' and second_token == 'which' and third_token == 'direction':
#         return 'inwhich'
#     elif first_token in ['is', 'was']:
#         return 'iss'
#     elif first_token in ['may', 'might']:
#         return 'may'
#     elif first_token == 'must':
#         return 'must'
#     elif first_token == 'should':
#         return 'should'
#     elif first_token == 'what':
#         return 'whats'
#     elif first_token == 'when' or (first_token == 'till' and second_token == 'when'):
#         return 'whens'
#     elif first_token == 'where':
#         return 'wheres'
#     elif first_token == 'which':
#         return 'whichs'
#     elif first_token in ['who', 'whom']:
#         return 'whos'
#     elif first_token == 'why':
#         return 'whys'
#     elif first_token in ['will', 'would']:
#         return 'wills'
    
#     return 'other'

def chunked(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]