// Generated GraphQL types, do not edit manually.

import * as Types from '../../graphql/types';

export type AssetCatalogTableQueryVariables = Types.Exact<{[key: string]: never}>;

export type AssetCatalogTableQuery = {
  __typename: 'DagitQuery';
  assetsOrError:
    | {
        __typename: 'AssetConnection';
        nodes: Array<{
          __typename: 'Asset';
          id: string;
          key: {__typename: 'AssetKey'; path: Array<string>};
          definition: {
            __typename: 'AssetNode';
            id: string;
            groupName: string | null;
            isSource: boolean;
            hasMaterializePermission: boolean;
            description: string | null;
            partitionDefinition: {__typename: 'PartitionDefinition'; description: string} | null;
            repository: {
              __typename: 'Repository';
              id: string;
              name: string;
              location: {__typename: 'RepositoryLocation'; id: string; name: string};
            };
          } | null;
        }>;
      }
    | {
        __typename: 'PythonError';
        message: string;
        stack: Array<string>;
        errorChain: Array<{
          __typename: 'ErrorChainLink';
          isExplicitLink: boolean;
          error: {__typename: 'PythonError'; message: string; stack: Array<string>};
        }>;
      };
};

export type AssetCatalogGroupTableQueryVariables = Types.Exact<{
  group?: Types.InputMaybe<Types.AssetGroupSelector>;
}>;

export type AssetCatalogGroupTableQuery = {
  __typename: 'DagitQuery';
  assetNodes: Array<{
    __typename: 'AssetNode';
    id: string;
    groupName: string | null;
    isSource: boolean;
    hasMaterializePermission: boolean;
    description: string | null;
    assetKey: {__typename: 'AssetKey'; path: Array<string>};
    partitionDefinition: {__typename: 'PartitionDefinition'; description: string} | null;
    repository: {
      __typename: 'Repository';
      id: string;
      name: string;
      location: {__typename: 'RepositoryLocation'; id: string; name: string};
    };
  }>;
};

export type AssetCatalogGroupTableNodeFragment = {
  __typename: 'AssetNode';
  id: string;
  groupName: string | null;
  isSource: boolean;
  hasMaterializePermission: boolean;
  description: string | null;
  assetKey: {__typename: 'AssetKey'; path: Array<string>};
  partitionDefinition: {__typename: 'PartitionDefinition'; description: string} | null;
  repository: {
    __typename: 'Repository';
    id: string;
    name: string;
    location: {__typename: 'RepositoryLocation'; id: string; name: string};
  };
};
