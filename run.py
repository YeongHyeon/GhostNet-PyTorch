import argparse, time, os, operator

import torch
import source.agent as agt
import source.utils as utils
import source.connector as con
import source.procedure as proc
import source.datamanager as dman

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    ngpu = FLAGS.ngpu
    if(not(torch.cuda.is_available())): ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataset = dman.DataSet()

    agent = agt.Agent(nn=con.connect(nn=FLAGS.nn), \
        dim_h=dataset.dim_h, dim_w=dataset.dim_w, dim_c=dataset.dim_c, num_class=dataset.num_class, \
        k_size1=FLAGS.k_size1, k_size2=FLAGS.k_size2, ratio=FLAGS.ratio, filters=FLAGS.filters, \
        learning_rate=FLAGS.lr, path_ckpt='Checkpoint', ngpu=ngpu, device=device)

    time_tr = time.time()
    proc.training(agent=agent, dataset=dataset, batch_size=FLAGS.batch, epochs=FLAGS.epochs)
    time_te = time.time()
    best_dict, num_model = proc.test(agent=agent, dataset=dataset)
    time_fin = time.time()

    tr_time = time_te - time_tr
    te_time = time_fin - time_te
    print("Time (TR): %.5f [sec]" %(tr_time))
    print("Time (TE): %.5f (%.5f [sec/sample])" %(te_time, te_time/num_model/dataset.num_te))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='')
    parser.add_argument('--ngpu', type=int, default=1, help='')

    parser.add_argument('--nn', type=int, default=0, help='')

    parser.add_argument('--k_size1', type=int, default=3, help='')
    parser.add_argument('--k_size2', type=int, default=3, help='')
    parser.add_argument('--ratio', type=int, default=2, help='')
    parser.add_argument('--filters', type=str, default="16,32,64", help='')
    parser.add_argument('--lr', type=float, default=1e-3, help='')

    parser.add_argument('--batch', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
