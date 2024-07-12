do:
pip install gym[toy_text]
conda install potassco::telingo
conda install -c potassco telingo

## ToDo / Status:
- Paper: https://www.overleaf.com/project/666d2ecdd9d2e0c3e9a800ac
- Note: add forster / forester paradox

- Implement Norms:
  - (Agent must move towards goal / don't go back / don't visit tile twice / follow planned or given path)
      -  Agent must reach goal is independent of this another norm
  - (If so, then re-return)
  - consider 'applied' paradoxes

- Define settings for evaluation:
  - scaled sum of rewards + violations (maybe split violations with different scales??)

- check planning with learning:
  - use more ways to handle to trigger planning (decaying, safe-init, eps-random)

- let traverser-paths be cyclic, and have a level where it jumps between both paths to block the agent
  - needs new config-param
- add steal-present with high value, but forbidden norm

- Implement plotting and output data:
  - need to check if norms are violated in the target policy, needs to be checked how to do so! 
