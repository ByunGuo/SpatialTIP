# Independent EDM adapter: original algorithm sources remain unchanged.
args <- commandArgs(trailingOnly=TRUE)
if (!(length(args) %in% c(4,5))) stop('Usage: run_reference_pilot.R ROOT INPUT OUT METHOD [smoke|pilot|formal]')
mode <- if(length(args)==5) args[5] else 'smoke'
if(!(mode %in% c('smoke','pilot','formal'))) stop('Invalid execution mode')
root <- normalizePath(args[1]); input <- normalizePath(args[2]); out <- args[3]; method <- args[4]
.libPaths(c(file.path(root,'experiments/edm/Rlib'), .libPaths()))
Sys.setenv(R_LIBS_USER=paste(.libPaths(),collapse=.Platform$path.sep))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
set.seed(42)
counts <- as.matrix(Matrix::readMM(file.path(input,'counts_spot_gene.mtx')))
normalized <- as.matrix(Matrix::readMM(file.path(input,'normalized_spot_gene.mtx')))
barcodes <- readLines(file.path(input,'barcodes.tsv')); genes <- readLines(file.path(input,'genes.tsv'))
stopifnot(identical(dim(counts),dim(normalized)),length(barcodes)==nrow(counts),length(genes)==ncol(counts),all(is.finite(counts)),all(counts>=0),all(counts==round(counts)))
if(mode=='smoke' && (nrow(counts)>128 || ncol(counts)>256)) stop('Smoke reference input limited to 128 spots and 256 genes')
write_matrix <- function(x,name) {
  stopifnot(identical(dim(x),dim(counts)))
  con <- file(file.path(out,paste0(name,'.f32')),'wb')
  writeBin(as.numeric(t(x)),con,size=4,endian='little'); close(con)
}
writeLines(barcodes,file.path(out,'barcodes.tsv')); writeLines(genes,file.path(out,'genes.tsv'))
if(method=='ALRA') {
  result <- ALRA::alra(normalized) # native automatic rank, q=10, quantile.prob=0.001
  write_matrix(result[[3]],'completed')
} else if(method=='ZINB_WaVE') {
  valid <- colSums(counts)>0
  y <- t(counts[,valid,drop=FALSE]); rownames(y)<-genes[valid];colnames(y)<-barcodes
  fit <- zinbwave::zinbFit(y,K=0,verbose=TRUE,BPPARAM=BiocParallel::SerialParam())
  # Posterior NB component weights, not prior Pi.
  weights <- zinbwave::computeObservationalWeights(fit,y)
  posterior <- matrix(NA_real_,nrow(counts),ncol(counts))
  posterior[,valid] <- t(1-weights)
  write_matrix(posterior,'dropout_probability')
  saveRDS(fit,file.path(out,'fit.rds'))
} else if(method=='scImpute') {
  y <- t(counts); rownames(y)<-genes;colnames(y)<-barcodes
  path <- file.path(out,'raw_count.rds');saveRDS(y,path)
  work <- paste0(normalizePath(out),'/')
  scImpute::scimpute(count_path=path,infile='rds',outfile='rds',type='count',out_dir=work,labeled=FALSE,drop_thre=0.5,Kcluster=10,ncores=5)
  clust <- readRDS(file.path(out,'clust.rds'))
  scale <- colSums(y);scale[scale==0]<-1
  logcounts <- log10(sweep(y,2,1e6/scale,FUN='*')+1.01)
  posterior <- matrix(NA_real_,nrow(y),ncol(y))
  for(cc in sort(unique(clust[!is.na(clust)]))) {
    cells <- which(clust==cc); pathpars<-file.path(out,paste0('pars',cc,'.rds'))
    if(length(cells)<=1 || !file.exists(pathpars)) next
    pars <- readRDS(pathpars)
    valid <- scImpute:::find_va_genes(pars,subcount=logcounts[,cells,drop=FALSE])
    if(length(valid)<=10) next
    for(g in valid) {
      p <- scImpute:::calculate_weight(logcounts[g,cells],pars[g,])[,1]
      p[logcounts[g,cells]>pars[g,'mu'] & p>0.5] <- 0
      p[!is.finite(p)] <- NA_real_
      posterior[g,cells] <- p
    }
  }
  write_matrix(t(posterior),'dropout_probability')
} else stop('Unknown method')
writeLines(c(paste('method',method),paste('version',as.character(packageVersion(if(method=='ZINB_WaVE') 'zinbwave' else method))),paste('mode',mode),paste('spots',nrow(counts)),paste('genes',ncol(counts)),'complete TRUE'),file.path(out,'status.txt'))
