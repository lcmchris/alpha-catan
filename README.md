# alpha-catan
alpha-catan is a fun endeavour to test out deep learning techniques.

I personally find Catan the most boring board game out there, most of the time I either feel helpless or crushing my opponents.
This extremity in the get go makes it appear like an 'easy' problem to solve with Deep learning.

I know there are other similar repos but nothing those didn't fully satisfy me, and why not try myself?

## Basic heuristics
I will skip over the basic rules which you can find [here|https://www.catan.com/sites/default/files/2021-06/catan_base_rules_2020_200707.pdf]


## Approach

### Simplistic view

My first thougth is to simplify the problem to just 1 player. How can this player maximise winning as fast as possbile without anyone against him?
This can help understand how the AI will value tiles and where they are located when there are no competition.


No trades - trades are slightly harder to model. An edge case is if this is only a 2 player game where both are optimally playing then no one actually trades as it is a zero sum game. I think for 3,4 player games, it will depend if there is any incentive to come 2nd or 3rd.

Theres actually not that many moves availble after the initial phase. On each turn, consider if only one move can be made, the max_moves = num_of_settlement * 3 + possible_settlement_points + num_of_settlement (option to upgrade to city) + 1 (option to buy a dev card). With only 2 settlements at the start and limited resources, this at most amounts to < 6 + 1 + 2 + 1 = 10

