function individualsBreed = selectFromPopulation(individualsBest,best_samples, lucky_few)

for i=1:best_samples
    individualsBreed(i) = individualsBest(i);
end

lf =1;
while lf <= lucky_few 
      r = randi([1 max(size(individualsBest))]);
      individualsBreed(lf+best_samples) = individualsBest(r);
    
    lf = lf +1;
end


individualsBreed(randperm(length(individualsBreed)))


end