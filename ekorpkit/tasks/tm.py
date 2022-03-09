# elif "export_samples" in task_args:
#     # tm = TopicModel(model_name=model_name, model_dir=model_dir, output_dir=args.output_dir)
#     # tm.load_model(model_file=args.model_file)
#     tm.export_samples(**cfg.task.export_samples)

# def export_corpus_samples(args, corpus_args, model_args):
#     os.makedirs(os.path.abspath(args.export_dir), exist_ok=True)

#     sample_ids = {}
#     sample_dfs = {}
#     for sample_type in args.sample_types:
#         filename = '{}-{}-{}.csv'.format(model_args.model_name, args.model_id, sample_type)
#         sample_id_path = f'{model_args.output_dir}/{model_args.model_name}/{filename}'
#         col_names = pd.read_csv(sample_id_path, index_col=0, nrows=0).columns
#         col_types = {col: str for col in col_names}
#         df_id = pd.read_csv(sample_id_path, index_col=None, dtype=col_types)
#         sample_ids[sample_type] = df_id
#         sample_dfs[sample_type] = []

#     corpus_paths = load_corpus_paths(corpus_args.corpus_dir, corpus_args.name,
#         corpus_type=corpus_args.corpus_type, corpus_filetype=corpus_args.corpus_filetype)
#     for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
#         print('-' * 120)
#         corpus = eKorpkit.load(corpus_name, corpus_dir=corpus_file,
#                                 corpus_type=corpus_args.corpus_type,
#                                 load_light=False,
#                                 load_dataframe=True, id_keys=corpus_args.id_keys)
#         for sample_type in args.sample_types:
#             df_id = sample_ids[sample_type]
#             df = df_id.query('corpus_filename.str.startswith(@corpus_name)', engine='python')
#             df = df.merge(corpus.docs, on=list(args.merge_cols))
#             df.text = df.text.str.replace('\n\n', '\n')
#             if len(df.index) > 0:
#                 df['corpus_filename'] = corpus_name
#                 df.rename(columns={'corpus_filename': 'corpus'}, inplace=True)
#                 print(df.head())
#                 sample_dfs[sample_type].append(df)

#     combined_dfs = []
#     for sample_type in args.sample_types:
#         df_exp = pd.concat(sample_dfs[sample_type])
#         print(df_exp.head())
#         filename = '{}-{}-{}-samples.csv'.format(model_args.model_name, args.model_id, sample_type)
#         export_path = f'{args.export_dir}/{filename}'
#         df_exp.to_csv(export_path, index=False)
#         filename = '{}-{}-{}-samples.json'.format(model_args.model_name, args.model_id, sample_type)
#         export_path = f'{args.export_dir}/{filename}'
#         df_exp.to_json(export_path, orient='records', force_ascii=False, indent=4)
#         df_exp['sample_type'] = sample_type
#         combined_dfs.append(df_exp)

#     combined_df = pd.concat(combined_dfs)
#     filename = '{}-{}-samples.csv'.format(model_args.model_name, args.model_id)
#     export_path = f'{args.export_dir}/{filename}'
#     combined_df.to_csv(export_path, index=False)
#     filename = '{}-{}-samples.json'.format(model_args.model_name, args.model_id)
#     export_path = f'{args.export_dir}/{filename}'
#     combined_df.to_json(export_path, orient='records', force_ascii=False, indent=4)
