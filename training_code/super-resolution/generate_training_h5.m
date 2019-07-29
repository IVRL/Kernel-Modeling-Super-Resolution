% original data folder
hrdir = '/scratch/mfr/DIV2K/DIV2K_train_HR/';
% configuration
p = 128;

% stride
stride = 16;

fnum = 1;
files = dir([hrdir '/' '*.png']);
savedir = 'train';

% 3 channels
h5create([savedir '/hr.h5'],  '/data', [p p 3 Inf], 'Datatype', 'single', 'ChunkSize', [p p 3 1]);

for k = 1:length(files)
    file = files(k).name;
    image = imread([hrdir '/' file]);
    image = im2double(image);
    
    [w h c] = size(image);
    i = 1;
    j = 1;
    
    while ((j + p <= h))
        hrpatch = image(i:i+p-1, j:j+p-1, :);
        
        h5write([savedir '/hr.h5'],  '/data', single(hrpatch),  [1 1 1 fnum], [p p 3 1]);
        
        fnum = fnum + 1;
        
        if (mod(fnum, 1000) == 0)
            sprintf('already generated %d patches\n', fnum)
        end
        
        j = j + stride;
        if (i + p > w)
            i = 1;
            j = j + stride;
        end
    end
end
sprintf('generated %d patches\n', fnum)
