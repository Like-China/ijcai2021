using ArgParse
using DataFrames
include("utils.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--datapath"
            arg_type=String
            default="C:/Users/likem/Desktop/t2vec-master/data/"
    end
    parse_args(s; as_symbols=true)
end

datapath = args[:datapath]

porto2h5("$datapath/porto.csv")
