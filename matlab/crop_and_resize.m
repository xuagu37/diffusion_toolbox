function crop_and_resize(fileNames)

fileNum = length(fileNames);

for i = 1:fileNum
    crop(fileNames{i})  
end
A = imread(fileNames{1});
[rowsA, colsA, ~] = size(A);
for i = 2:fileNum
    B = imread(fileNames{i});
    B = imresize(B, [rowsA colsA]);
    imwrite(B,fileNames{i});    
end

end
