%% 텍스트 파일에서 데이터를 가져옴
% 다음 텍스트 파일에서 데이터를 가져오기 위한 스크립트:
%
%    파일 이름: /home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/log.txt
%
% MATLAB에서 2019-10-04 10:28:52에 자동 생성됨

fname = ["/home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/log.txt",...
    "/home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/restart_50/log.txt",...
    "/home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/restart_50/restart_90/log.txt",...
    "/home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/restart_50/restart_90/restart_140/log.txt",...
    "/home/hayoung/Data_sync/Dropbox/packages/08.OpenMDAO_OpenLSTO/LSTO_DiscreteAdjoint/LSTO_AD/save_new/restart_50/restart_90/restart_140/restart_205/log.txt"];

num_f =length(fname);
%% 가져오기 옵션을 설정하고 데이터 가져오기
opts = delimitedTextImportOptions("NumVariables", 4);

% 범위 및 구분 기호 지정
opts.DataLines = [1, Inf];
opts.Delimiter = ",";

% 열 이름과 유형 지정
opts.VariableNames = ["f1", "f2", "ft", "g"];
opts.VariableTypes = ["double", "double", "double", "double"];

% 파일 수준 속성 지정
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

%% 출력 유형으로 변환
f1 = zeros(205,1); 
f2 = zeros(205,1); 
ft = zeros(205,1); 
g = zeros(205,1);

id = [1, 50, 90, 140, 205, 300];

for nn = 1:4
    % 데이터 가져오기
    tbl = readtable(fname(nn), opts);

    f1(id(nn):id(nn+1)) = tbl.f1(1:1-id(nn)+id(nn+1));
    f2(id(nn):id(nn+1)) = tbl.f2(1:1-id(nn)+id(nn+1));
    ft(id(nn):id(nn+1)) = tbl.ft(1:1-id(nn)+id(nn+1));
    g(id(nn):id(nn+1)) = tbl.g(1:1-id(nn)+id(nn+1));
end


%% 임시 변수 지우기
clear opts tbl