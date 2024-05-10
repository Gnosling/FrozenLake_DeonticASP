do:
pip install gym[toy_text]
conda install potassco::telingo
conda install -c potassco telingo

## ToDo / Status:
- Implement Norms:
  - Agent must not be on the same tile as bot
  - If so, then go straight
  - Agent must reach goal
  - (Agent must move towards goal / don't go back / don't visit tile twice / follow planned or given path)
  - (If so, then re-return)

- Should agent know 'cracked' tiles or assume everything is frozen?
  - only if it affects planned rewards?

- Write proposal until 19th June:
  - Intro
  - stateOfArt
  - Goal / research questions
  - Method
  - Connections to L&C (ie. state lectures
  - literature!!
- Define settings for evaluation:
  - only rewards
  - only obligations
  - first rewards than obligations
  - first obligations than rewards