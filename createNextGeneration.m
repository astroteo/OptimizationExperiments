
function nextGeneration = createNextGeneration(individuals,best_samples,lucky_few,n_children,chance_of_mutation)

    populationSorted = computePerfPopulation(individuals);
    %disp('OK1')
    %size(populationSorted)
	nextBreeders = selectFromPopulation(populationSorted,individuals,best_samples, lucky_few);
    %disp('OK2')
    %size(nextBreeders)
	nextPopulation = createChildren(nextBreeders, n_children);
    %disp('OK3')
    %size(nextPopulation)
	nextGeneration = mutatePopulation(nextPopulation, chance_of_mutation);
    %disp('OK4')
    %size(nextGeneration)
    
   

end


function individualsBreed = selectFromPopulation(individualsBest,individuals,best_samples, lucky_few)
    

    global L V
    for i=1:best_samples
        individualsBreed(i) = individualsBest(i);
    end

    lf =1;
    while lf <= lucky_few 
          r = randi([1 max(size(individualsBest))]);
          individualsBreed(lf+best_samples) = individuals(r);

        lf = lf +1;
    end


    individualsBreed(randperm(length(individualsBreed)));
    %size(individualsBreed)
    
    for ib = individualsBreed
        
        ib.r0 = (-1 + 2 * rand ) * L/4 + ib.r0;
        ib.v0 = (-1 + 2* rand ) * V/4 + ib.v0;
    end


end


function individualsChildren = createChildren(individualsBreed, n_children)
    
    n_couples =max(size(individualsBreed))/2;
    k=1;
    for i=1:n_couples
        for j =1:n_children
            parent1 = individualsBreed(i);
            parent2 = individualsBreed(n_couples*2-i);

           individualChild =  createChild(parent1,parent2);
           individualsChildren(k) = individualChild;
           k=k+1;

        end
    end
    
    disp(size(individualsChildren))
end

function individualChild = createChild(individual1,individual2)
    
    individualChild = Individual;

    individualChild.r0 = individual1.r0;
    individualChild.v0 = individual1.v0;

    individualChild.score =0;

   
    individualChild.biasHiddenLayer1 = 0.5 .* (individual1.biasHiddenLayer1 + individual2.biasHiddenLayer1);
    individualChild.biasHiddenLayer2 = 0.5.* (individual1.biasHiddenLayer2 + individual2.biasHiddenLayer2);
    
     %AVERAGE BETWEEN NEURONS ==> BAD CHOICE!!!!
     %individualChild.hiddenLayer2 = 0.5 .* (individual1.hiddenLayer2 + individual2.hiddenLayer2);
     %individualChild.hiddenLayer1 = 0.5 .* (individual1.hiddenLayer1 + individual2.hiddenLayer1);
     
    %half neurons from father, half neurons from mother
    n_hidden1 = max(size(individual1.hiddenLayer1));
    n_hidden2 = max(size(individual1.hiddenLayer2));
    
    
    
    childHiddenLayer1 = zeros(n_hidden1,4);
    childHiddenLayer2 = zeros(2,n_hidden2);
   
    hL1 = randi([0 1], n_hidden1,4);
    hL2 = randi([0 1], 2,n_hidden2);
    
    for i=1:n_hidden1
        for j=1:4
            if hL1(i,j) ==1
                childHiddenLayer1(i,j) = individual1.hiddenLayer1(i,j);
            else
                childHiddenLayer1(i,j) = individual2.hiddenLayer1(i,j);
            end
            
        end
    end
    
    
    for i=1:2
        for j=1:n_hidden2
            if hL2(i,j) ==1
                childHiddenLayer2(i,j) = individual1.hiddenLayer2(i,j);
            else
                childHiddenLayer2(i,j) = individual2.hiddenLayer2(i,j);
            end
            
        end
    end
    
    
    individualChild.hiddenLayer1 = childHiddenLayer1;
    individualChild.hiddenLayer2 = childHiddenLayer2;
    
    
    %disp('CHILD')
    %size(individualChild.hiddenLayer1)
    %individualChild.hiddenLayer1
    %size(individualChild.biasHiddenLayer1)
    %individualChild.biasHiddenLayer1
    %size(individualChild.hiddenLayer2)
    %individualChild.hiddenLayer2 
    %size(individualChild.biasHiddenLayer2)
    %individualChild.biasHiddenLayer2 
    
    

end


function individualsMutated = mutatePopulation(individuals , chance_of_mutation)

        for i =1:max(size(individuals))
            
            if rand * 100 > chance_of_mutation %setted to 5% like
                
                individualMutated = mutateIndividual(individuals(i));
                individualsMutated(i) = individualMutated;
                
            else
                
                individualsMutated(i) = individuals(i);
                
                
            end
            
            
            
        end

end

function individualMutated = mutateIndividual(individual)

    individualMutated = Individual;
    
    n_hidden1 = max(size(individual.hiddenLayer1));
    n_hidden2 = max(size(individual.hiddenLayer2));
    
    
    
    individualMutated.score =0;
    individualMutated.traj = [individual.r0',0,individual.v0',0];
    individualMutated.r0 = individual.r0;
    individualMutated.v0 =individual.v0;
    
    %disp('WWWWTTTFFF')
    %size(individual.hiddenLayer1)
    
    individualMutated.hiddenLayer1 = individual.hiddenLayer1 + 0.0001 * (rand(n_hidden1,4) *2 - ones(n_hidden1,4)) ;
    individualMutated.biasHiddenLayer1 =  individual.biasHiddenLayer1;
    individualMutated.hiddenLayer2 = individual.hiddenLayer2 + 0.0001 * (rand(2,n_hidden2) *2 - ones(2,n_hidden2));
    individualMutated.biasHiddenLayer2 =individual.biasHiddenLayer2;
    
    %disp('MUTATION')
    %size(individualMutated.hiddenLayer1)
    %individualMutated.hiddenLayer1
    %individualMutated.biasHiddenLayer1
    %size(individualMutated.hiddenLayer2)
    %individualMutated.hiddenLayer2
    %individualMutated.biasHiddenLayer2
end

