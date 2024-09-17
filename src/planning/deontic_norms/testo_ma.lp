% python -m telingo --opt-mode=opt --quiet=1 testo_ma.lp


% Define some facts
person(john).
person(mary).
person(susan).

% python -m telingo --opt-mode=opt --quiet=1 testo_ma.lp

#show job/1.

% Assign one person to do the job
{job(X) : person(X)} = 1.

% Weak constraint to prefer assigning 'john'
:~ not job(john). [1@2]

% Weak constraint to prefer assigning 'mary', higher prio is higher class
:~ not job(mary). [1@3]