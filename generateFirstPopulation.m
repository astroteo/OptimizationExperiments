function individuals = generateFirstPopulation(n_individuals, n_hidden1,n_hidden2,interval,r0,v0)

 for i=1:n_individuals
     
     individual = Individual;
     
     if interval(1) == -1
        hiddenLayer1 = rand(n_hidden1,4) *2 - ones(n_hidden1,4);
        biasHiddenLayer1 = ones(n_hidden1,1);

        hiddenLayer2 = rand(2,n_hidden2) * 2 - ones(2,n_hidden2);
        biasHiddenLayer2 = ones(2,1);
        
    else % weights between 0,1
        hiddenLayer1 = rand(n_hidden1,4);
        biasHiddenLayer1 = zeros(n_hidden1,1);

        hiddenLayer2 = rand(2,n_hidden2);
        biasHiddenLayer2 = zeros(2,1);
     end
     
     individual.hiddenLayer1 = hiddenLayer1;
     individual.hiddenLayer2 = hiddenLayer2;
     individual.biasHiddenLayer1 = biasHiddenLayer1;
     individual.biasHiddenLayer2 = biasHiddenLayer2;
     individual.score =0;
     individual.traj = [r0',0,v0',0];
     individual.r0 = r0;
     individual.v0 = v0;
     
     individuals(i) = individual;
     
            
 

 end





end


