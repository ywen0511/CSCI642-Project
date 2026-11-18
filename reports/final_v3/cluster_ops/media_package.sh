#!/bin/bash
# media_package.sh <analysis_name> <media_name> <pkg_prefix>: media tarball + per-GIF assets, each dir with SHA256SUMS
C=/data/rtxw01/nbak/robot_intel/kztrgg/projs/code/csci642-v3; A=$1 M=$2 P=$3
rm -rf $C/packages/${P}_media $C/packages/${P}_gif; mkdir -p $C/packages/${P}_media $C/packages/${P}_gif
tar -C $C -czf $C/packages/${P}_media/analysis_and_media.tar.gz analysis/$A media/$M
cp $C/media/$M/gif/*.gif $C/packages/${P}_gif/
(cd $C/packages/${P}_media && sha256sum * > SHA256SUMS); (cd $C/packages/${P}_gif && sha256sum * > SHA256SUMS)
echo "MEDIA_PKG $P $(du -sh $C/packages/${P}_media $C/packages/${P}_gif | cut -f1 | tr "\n" " ")"
