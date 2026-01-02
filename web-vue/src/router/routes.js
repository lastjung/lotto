const routes = [
  {
    path: '/',
    component: () => import('layouts/MainLayout.vue'),
    children: [
      { path: '', component: () => import('pages/IndexPage.vue') },
      { path: 'history', component: () => import('pages/HistoryPage.vue') },
      { path: 'models', component: () => import('pages/ModelsPage.vue') },
      { path: 'system', component: () => import('pages/SystemPage.vue') }
    ]
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: '/:catchAll(.*)*',
    component: () => import('pages/ErrorNotFound.vue')
  }
]

export default routes
