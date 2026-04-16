library(HIBAG)
args <- commandArgs(trailingOnly = TRUE)
in_file = args[1]
ref_file = args[2]
out_file = args[3]

HLA = c("A", "B", "C", "DRB1", "DQA1", "DQB1", "DPB1")
hla_hibag = function(in_file, ref_file, out_file){
    model.list <- get(load(ref_file))
    dataset = hlaBED2Geno(bed.fn=paste0(in_file, ".bed"), fam.fn=paste0(in_file, ".fam"), bim.fn=paste0(in_file, ".bim"))

    df_list = list()
    for (locus in HLA) {
        model = hlaModelFromObj(model.list[[locus]])
        hla_pred = hlaPredict(model, dataset, type="response+prob")
        pred_df = as.data.frame(hla_pred$value)
        locus_name = paste0("HLA-", locus)
        pred_df['HLA'] = locus_name
        df_list[[locus_name]] = pred_df
    }
    df = do.call(rbind, df_list)
    ref = strsplit(ref_file, split="-HLA")[[1]][1]
    write.table(df, file=paste0(out_file, ".txt"), sep="\t", quote=F, row.names=F)
}
hla_hibag(in_file, ref_file, out_file)
